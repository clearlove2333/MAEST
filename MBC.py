import logging
import dgl
import numpy as np
from tqdm import tqdm
import torch
from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)

from models import build_model
from process.function import Function
from pathlib import Path
import os
import scanpy as sc
import matplotlib.pyplot as plt


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print("所用GPU为:",device)
    seeds = args.seeds
    dataset = args.dataset
    max_epoch = args.max_epoch
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    # save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler
    samples = args.sample
    power = args.power

    seeds_ari_list = []
    # 遍历seed
    for i, seed in enumerate(seeds):

        result_path = Path('./results/' + f'{dataset}/')
        result_path.mkdir(parents=True, exist_ok=True)
        #初始化模型
        function = Function(datatype=dataset, n_clusters=10)
        adata = function.loadData()
        adata, u, v = function.process(adata)
        #创建dgl数据
        graph = dgl.graph((torch.tensor(u), torch.tensor(v)))
        if(args.self_loop):
            graph = dgl.add_self_loop(graph)
        x = torch.tensor(adata.obsm['feat'])
        args.num_features = x.shape[1]

        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        # 创建模型
        model = build_model(args)
        model.to(device)

        #加载模型
        if args.load_model:
            model.load_state_dict(torch.load("./saved_models/MAEST_" + args.dataset + ".pt"))
            ari=function.clusting(adata, model, graph, x, args.power, device)
            print(ari)
            return

        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        # if args.wandb:
        #     if not os.path.exists("./wandb/"):
        #         os.makedirs("./wandb")

        #     wandb.init(config=args,
        #             project="MAEST",
        #             name="baseline_{}".format(dataset),
        #             dir="./wandb/",
        #             job_type="training",
        #             reinit=True)
        
        logging.info("start training..")
        best_ari = 0
        graph = graph.to(args.device)
        x = x.to(device)

        # target_nodes = torch.arange(x.shape[0], device=x.device, dtype=torch.long)
        epoch_iter = tqdm(range(max_epoch))
        for epoch in epoch_iter:
        # for epoch in range(max_epoch):
            model.train()
            loss = model(graph, x)

            loss_dict = {"loss": loss.item()}
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
            if logger is not None:
                loss_dict["lr"] = get_current_lr(optimizer)
                logger.note(loss_dict, step=epoch)

            if (epoch + 1) % 50 == 0:
                #聚类
                function.clusting_no_label(adata, model, graph, x, power, device)
                fig, ax_list = plt.subplots(1, 2, figsize=(10, 4))
                # fig, ax_list = plt.subplots(1, 2, figsize=(12, 5))
                sc.pl.embedding(adata_section1,
                               basis='spatial',
                               color='domain',
                               show = False,
                               s=100,
                               title='Section 1',
                               ax = ax_list[0])

                sc.pl.embedding(adata_section2,
                               basis='spatial',
                               color='domain',
                               show = False,
                               s=100,
                               title = ['Section 2'],
                               ax = ax_list[1])

                plt.tight_layout(w_pad=0.2)   
                plt.savefig("./results/" + dataset + "/" + str(epoch) + ".png")
                # adata.write("./results/" + dataset + "/" + str(epoch) +".h5ad")
                # return

        if logger is not None:
            logger.finish()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args)
    print(args)
    main(args)
