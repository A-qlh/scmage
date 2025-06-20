import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import random
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from datasets import Loader, apply_noise
from model import AutoEncoder
from evaluate import evaluate
from util import AverageMeter

def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def inference(net, data_loader_test):
    net.eval()
    feature_vector = []
    labels_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            # feature_vector.extend(net.feature(x.cuda()).detach().cpu().numpy())
            feature_vector.extend(net.feature(x.to(device)).detach().cpu().numpy())
            labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    dis = []
    resolutions = sorted(list(np.arange(0.01, 2.5, increment)), reverse=True)
    i = 0
    res_new = []
    for res in resolutions:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(
            adata.obs['leiden']).leiden.unique())
        dis.append(abs(count_unique_leiden-fixed_clus_count))
        res_new.append(res)
        if count_unique_leiden == fixed_clus_count:
            break
    reso = resolutions[np.argmin(dis)]

    return reso


def train(args):
    data_load = Loader(args, dataset_name=args["dataset"], drop_last=True)
    data_loader = data_load.train_loader
    data_loader_test = data_load.test_loader
    x_shape = args["data_dim"]

    results = []

    # Hyper-params
    init_lr = args["learning_rate"]
    max_epochs = args["epochs"]
    mask_probas = [0.4] * x_shape
    noise_sd = args.get("noise_sd", 0.2)

    # setup model
    model = AutoEncoder(
        num_genes=x_shape,
        hidden_size=128,
        num_attention_heads=2,
        attention_hidden_size=128,
        masked_data_weight=0.75,
        mask_loss_weight=0.7,
        noise_sd=args.get("noise_sd", 0.2)
    ).to(device)

    model_checkpoint = 'model_checkpoint.pth'

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # train model
    for epoch in range(max_epochs):
        model.train()
        meter = AverageMeter()
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            x_corrupted, mask = apply_noise(x, mask_probas)

            optimizer.zero_grad()
            x_corrupted_latent, loss_ae = model.loss_mask(
                x_corrupted, x, mask, add_noise=True
            )
            loss_ae.backward()
            optimizer.step()
            meter.update(loss_ae.detach().cpu().numpy())

        if epoch == 60:
            # Generator in eval mode
            latent, true_label = inference(model, data_loader_test)
            if latent.shape[0] < 10000:
                clustering_model = KMeans(n_clusters=args["n_classes"])
                clustering_model.fit(latent)
                pred_label = clustering_model.labels_
            else:
                adata = sc.AnnData(latent)
                sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
                reso = res_search_fixed_clus(adata, args["n_classes"])
                sc.tl.leiden(adata, resolution=reso)
                pred = adata.obs['leiden'].to_list()
                pred_label = [int(x) for x in pred]

            nmi, ari,ami, acc = evaluate(true_label, pred_label)
            ss = silhouette_score(latent, pred_label)

            res = {
                "nmi": round(nmi, 4),
                "ari": round(ari, 4),
                "ami": round(ari, 4),
                "acc": round(acc, 4),
                "sil": round(ss, 4),
                "dataset": args["dataset"],
                "epoch": epoch
            }

            results.append(res)

            print(f"\tEvaluate: [nmi: {nmi:.4f}] [ari: {ari:.4f}][ami: {ari:.4f}] [acc: {acc:.4f}] [sil: {ss:.4f}]")
            latent_df = pd.DataFrame(latent)
            latent_df.to_csv(args["save_path"] + "/embedding_" + str(epoch) + ".csv", index=False)
            pd.DataFrame({"True": true_label, "Pred": pred_label}).to_csv(
                args["save_path"] + "/types_" + str(epoch) + ".csv", index=False)

    torch.save({
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict()
    }, model_checkpoint)

    return results



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置固定的种子
    seed = 42
    set_seed(seed)

    for i in range(1):
        seed = random.randint(1, 100)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        args = {}
        args["num_workers"] = 4
        args["paths"] = {"data":"data",
                        "results":"res"}
        args['batch_size'] = 256
        args["data_dim"] = 1000
        args['n_classes'] = 4
        args['epochs'] = 100
        args["dataset"] = "Guo"
        args["learning_rate"] = 1e-3
        args["latent_dim"] = 32
        args["noise_sd"] = 0.6


        print(args)

        path = args["paths"]["data"]
        files = ["Guo"]

        results = pd.DataFrame()
        save_dir = make_dir(args["paths"]["results"], "a_summary")
        for dataset in files:
            print(f">> {dataset}")
            args["dataset"] = dataset
            args["save_path"] = make_dir(r"data", dataset)

            res = train(args)
            print(res)
            results = results.append(res)
            results.to_csv(args["paths"]["results"] +
                        "/res_all_data_test"+str(i)+".csv", header=True)