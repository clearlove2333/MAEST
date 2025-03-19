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


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    print("The GPU used is:",device)
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
    logs = args.logging
    use_scheduler = args.scheduler
    samples = args.sample

    seeds_ari_list = []

    for i, seed in enumerate(seeds):

        samples_ari_list = []
        for i, sample in enumerate(samples):

            function = Function(sample,dataset)
            adata = function.loadData()
            adata, u, v = function.process(adata)

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


            model = build_model(args)
            model.to(device)
            #power
            if sample in ['151669', '151670', '151671', '151672']:
                power = 3
            else:
                power = 0


            # args.load_model = True
            if args.load_model:
                model.load_state_dict(torch.load("./saved_models/MAEST_" + args.dataset + sample + ".pt"))
                # ari=function.clusting(adata, model, graph, x, power, device, refinement=True)
                # samples_ari_list.append(ari)
                # print(f'{sample}:{ari}')
                # umap_path='/zhupengfei/STExperiment/MAEST/results/DLPFC/'
                # function.clustingAndPlot(adata, model, graph, x, power, device, sample, umap_path, refinement=True)
                h5ad_path='/zhupengfei/STExperiment/MAEST/plot/denoise/'
                function.clustingAndSave(adata, model, graph, x, power, device, sample, h5ad_path, refinement=True)
                continue

            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler:
                logging.info("Use schedular")
                scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None
           
            logging.info("start training..")
            best_ari = 0
            graph = graph.to(args.device)
            x = x.to(device)

            result_path = Path('./results/' + f'DLPFC/{sample}/')
            result_path.mkdir(parents=True, exist_ok=True)

            epoch_iter = tqdm(range(max_epoch))
            for epoch in epoch_iter:
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

                if (epoch + 1) % 100 == 0:
                    ari=function.clusting(adata, model, graph, x, power, device, refinement=True)
                    if ari > best_ari:
                        best_ari = ari

            best_ari = round(best_ari, 2)
            samples_ari_list.append(best_ari)
            if logger is not None:
                logger.finish()

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args)
    print(args)
    main(args)
