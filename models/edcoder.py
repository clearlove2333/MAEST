from itertools import chain

from typing import Optional
import torch
import torch.nn as nn
from functools import partial

from .gat import GAT

from .loss_func import sce_loss

import copy
import random

import dgl.function as fn
import torch.nn.functional as F

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True, **kwargs) -> nn.Module:
    if m_type in ("gat", "tsgat"):
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            **kwargs,
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            num_dec_layers: int,
            num_remasking: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            remask_rate: float = 0.5,
            remask_method: str = "random",
            mask_method: str = "random",
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "byol",
            drop_edge_rate: float = 0.0,
            alpha_l: float = 2,
            lam: float = 1.0,
            bet: float = 1.0,
            delayed_ema_epoch: int = 0,
            momentum: float = 0.996,
            replace_rate: float = 0.0,
            zero_init: bool = False,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._mask_method = mask_method
        self._alpha_l = alpha_l
        self._delayed_ema_epoch = delayed_ema_epoch

        self.num_remasking = num_remasking
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._momentum = momentum
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method

        self._token_rate = 1 - self._replace_rate
        self._lam = lam
        self._bet = bet

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead if decoder_type in ("gat",) else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        if not zero_init:
            self.reset_parameters_for_token()

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        self.projector_ema = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )

        self.predictor = nn.Sequential(
            nn.PReLU(),
            nn.Linear(num_hidden, num_hidden)
        )

        # 测试修改掩码模块的project
        # self.predictor = nn.Sequential(
        #     nn.PReLU(),
        #     nn.Linear(num_hidden, 3000),
        #     nn.PReLU(),
        #     nn.Linear(3000, 3000)
        # )

        self.encoder_ema = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()

        self.print_num_parameters()

        self.discrimination_loss = nn.BCEWithLogitsLoss()

        self.DGI_projector = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.PReLU(),
            nn.Linear(num_hidden, 128),
            nn.PReLU(),
            nn.Linear(128, 20),
        )

        # self.DGI_projector = nn.Sequential(
        #     nn.Linear(num_hidden, num_hidden),
        # )

    def print_num_parameters(self):
        num_encoder_params = [p.numel() for p in self.encoder.parameters() if p.requires_grad]
        num_decoder_params = [p.numel() for p in self.decoder.parameters() if p.requires_grad]
        num_params = [p.numel() for p in self.parameters() if p.requires_grad]

        print(
            f"num_encoder_params: {sum(num_encoder_params)}, num_decoder_params: {sum(num_decoder_params)}, num_params_in_total: {sum(num_params)}")

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            print(f"=== Use mse_loss ===")
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            print(f"=== Use sce_loss and alpha_l={alpha_l} ===")
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, g, x, targets=None, epoch=0, drop_g1=None, drop_g2=None):  # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x, targets, epoch, drop_g1, drop_g2)
        return loss

    def mask_attr_prediction(self, g, x, targets, epoch, drop_g1=None, drop_g2=None):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = drop_g1 if drop_g1 is not None else g
    
        enc_rep = self.encoder(use_g, use_x,)
    
        with torch.no_grad():
            drop_g2 = drop_g2 if drop_g2 is not None else g
            #encoder_ema是和encoder结构一样的编码器
            #未掩码数据作为对比
            latent_target = self.encoder_ema(drop_g2, x,)
    
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                #projector_ema一层全连接的自编码器
                latent_target = self.projector_ema(latent_target[keep_nodes])
    
        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            #修改为mask_nodes
            # latent_pred = self.projector(enc_rep[mask_nodes])
            latent_pred = self.predictor(latent_pred)
            # 原loss
            loss_latent = sce_loss(latent_pred, latent_target, 1)
            # 测试修改掩码模块的project
            # loss_latent = sce_loss(latent_pred, x[keep_nodes], 1)
            # # 测试修改掩码模块的project
            # loss_latent = sce_loss(latent_pred, x[mask_nodes], 1)
    
        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)
    
        loss_rec_all = 0
        if self._remask_method == "random":
            #尝试不同大小的重掩码
            # muti_remask_rate = 0.5
            for i in range(self._num_remasking):
                #_num_remasking：重掩码的层数
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, self._remask_rate)
                # rep, remask_nodes, rekeep_nodes = self.random_remask(use_g, rep, muti_remask_rate)
                # muti_remask_rate = muti_remask_rate + 0
                recon = self.decoder(pre_use_g, rep)
    
                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(g, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError
        
        DGI_loss = self.DGI(g, x)
    
        loss = loss_rec + self._lam * loss_latent + self._bet * DGI_loss

        # loss = loss_rec + self._lam * loss_latent
    
        if epoch >= self._delayed_ema_epoch:
            self.ema_update()
        return loss

    def DGI(self, g, x):
        '''使用Dink-net DGI降维'''

        # augmentations
        x_aug = aug_feature_dropout(x, drop_rate=0.2).squeeze(0)

        # 部分替换负样本
        x_negative = self.encoding_mask_negative(g, x_aug, keep_rate_negative=0)


        enc_rep = self.encoder(g, x_aug, )
        enc_rep_negative = self.encoder(g, x_negative, )

        latent = self.DGI_projector(enc_rep)
        latent_negative = self.DGI_projector(enc_rep_negative)
        logit = torch.cat((latent.sum(1), latent_negative.sum(1)), 0)

        # label of discriminative task
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)

        # discrimination loss
        loss_disc = self.discrimination_loss(logit, disc_y)

        return loss_disc

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
                # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)
        # update(self.encoder, self.encoder_ema_negative)
        # update(self.projector, self.projector_ema_negative)

    def embed(self, g, x):
        rep = self.encoder(g, x)
        # #正则化
        # rep = F.normalize(rep, p=2, dim=1)
        return rep
    
    def embed_power(self, g, x, power):
        #多跳维度
        '''计算encoder和encoder+power层后的聚合特征相加'''
        local_h = self.encoder(g,x)
        #squeeze只能去除维度为1的维度
        feat = local_h.clone().squeeze(0)

        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).unsqueeze(1).to(local_h.device)
        for i in range(power):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'), fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        global_h = feat.unsqueeze(0)
        #从计算图分离，不计算梯度
        local_h, global_h = map(lambda tmp: tmp.detach(), [local_h, global_h])

        h = local_h + global_h
        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)

        return h

    def recon(self, g, x):
        rep = self.encoder(g, x)
        origin_rep = self.encoder_to_decoder(rep)
        recon = self.decoder(g, origin_rep)
        return recon


    def get_encoder(self):
        # self.encoder.reset_classifier(out_size)
        return self.encoder

    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # exclude isolated nodes
        # isolated_nodes = torch.where(g.in_degrees() <= 1)[0]
        # mask_nodes = perm[: num_mask_nodes]
        # mask_nodes = torch.index_fill(torch.full((num_nodes,), False, device=device), 0, mask_nodes, True)
        # mask_nodes[isolated_nodes] = False
        # keep_nodes = torch.where(~mask_nodes)[0]
        # mask_nodes = torch.where(mask_nodes)[0]
        # num_mask_nodes = mask_nodes.shape[0]

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    # def encoding_mask_negative(self, g, x, mask_nodes, keep_nodes, keep_rate_negative=0):

    #     num_nodes = g.num_nodes()
    #     num_keep_nodes = int(keep_rate_negative * num_nodes)

    #     # 正样本点
    #     keep_nodes_negative = keep_nodes[: num_keep_nodes]
    #     # 负样本点
    #     replace_nodes_negative = torch.cat((mask_nodes, keep_nodes[num_keep_nodes:]), dim=0)
    #     # 替换节点洗牌
    #     replace_nodes_perm = torch.randperm(num_nodes, device=x.device)[:replace_nodes_negative.shape[0]]

    #     out_x_negative = x.clone()
    #     out_x_negative[replace_nodes_negative] = x[replace_nodes_perm]
    #     out_x_negative[replace_nodes_negative] += self.enc_mask_token

    #     return out_x_negative, (replace_nodes_negative, keep_nodes_negative)

    def encoding_mask_negative(self, g, x, keep_rate_negative=0):
        '''选择一定比例洗牌作为负样本'''
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_keep_nodes = int(keep_rate_negative * num_nodes)
        # keep_nodes = perm[: num_keep_nodes]
        replace_nodes = perm[num_keep_nodes :]
        
        # 替换节点洗牌
        replace_nodes_perm = torch.randperm(num_nodes, device=x.device)[:replace_nodes.shape[0]]

        out_x_negative = x.clone()
        out_x_negative[replace_nodes] = x[replace_nodes_perm]
        out_x_negative[replace_nodes] += self.enc_mask_token

        return out_x_negative

    def random_remask(self, g, rep, remask_rate=0.5):

        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep

def aug_feature_dropout(input_feat, drop_rate=0.2):
    """
    dropout features for augmentation.
    args:
        input_feat: input features
        drop_rate: dropout rate
    returns:
        aug_input_feat: augmented features
    """
    aug_input_feat = copy.deepcopy(input_feat).squeeze(0)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    aug_input_feat = aug_input_feat.unsqueeze(0)

    return aug_input_feat