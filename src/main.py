import os
import pickle
from typing import Dict, List, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score
from torch.autograd import Variable
from torch.utils.data import DataLoader

from loss import LossHandler
from model import Model
from util.plot import (
    plot_per_bin_category_scores,
    plot_per_category_hist,
    plot_per_category_scores,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_lib(cfg: DictConfig) -> None:
    np.random.seed(cfg.const.seed)
    torch.manual_seed(cfg.const.seed)
    torch.cuda.manual_seed(cfg.const.seed)
    mlflow.set_tracking_uri(cfg.path.folder.mlruns)


def train(model: nn.Module, loader: DataLoader, loss_fn: LossHandler, opt: optim.Optimizer) -> None:
    model.train()
    total_loss = 0.0
    for _, batch in enumerate(loader):
        model.zero_grad()
        batch = [Variable(x).to(DEVICE) for x in batch]
        (cat_seq, y, hour_seq, uid, sess_len, region_seq, cat_candi, _, time_diff_seq, gps_cnt, checkin_cnt) = batch

        pred, pred_n = model(cat_seq, hour_seq, region_seq, time_diff_seq, uid, gps_cnt, checkin_cnt, sess_len)

        loss = loss_fn(pred, pred_n, y, cat_candi)
        loss.backward()
        opt.step()
        total_loss += float(loss)

    return


def test(model: nn.Module, loader: DataLoader, cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = torch.tensor([]).to(DEVICE)
    preds = torch.tensor([]).to(DEVICE)
    for _, batch in enumerate(loader):
        model.zero_grad()
        batch = [Variable(x).to(DEVICE) for x in batch]
        (cat_seq, y, hour_seq, uid, sess_len, region_seq, _, label, time_diff_seq, gps_cnt, checkin_cnt) = batch

        pred, _ = model(cat_seq, hour_seq, region_seq, time_diff_seq, uid, gps_cnt, checkin_cnt, sess_len)
        y = torch.where(y == -1, cfg.const.vocab.cat, y)

        for i in range(cat_seq.size(0)):
            if label[i] == 1:
                y_i, pred_i = y[i], pred[i]
                ys = torch.cat((ys, y_i.unsqueeze(0)))
                _, pred_i_sort = torch.sort(pred_i.data, descending=True)
                preds = torch.cat((preds, pred_i_sort[0].unsqueeze(0)))

    return ys.detach().cpu().numpy(), preds.detach().cpu().numpy()


def evaluate(y: np.ndarray, pred: np.ndarray, cfg: DictConfig) -> List[float]:
    y_bin = np.where(y == cfg.const.vocab.cat, 0, 1)
    pred_bin = np.where(pred == cfg.const.vocab.cat, 0, 1)

    rec_all = recall_score(y_bin, pred_bin, average="micro", zero_division=0)

    prec_per_cat = precision_score(y_bin, pred_bin, labels=[1, 0], average=None, zero_division=0)
    rec_per_cat = recall_score(y_bin, pred_bin, labels=[1, 0], average=None, zero_division=0)

    return [rec_all, rec_per_cat[0], prec_per_cat[0], rec_per_cat[1], prec_per_cat[1]]


def get_catind_catname(cfg: DictConfig) -> Dict[int, str]:
    df_checkin = pd.read_csv(cfg.path.file.checkin)
    df_gps = pd.read_csv(cfg.path.file.gps)
    df_gps_label = df_gps[(df_gps["label"] == 1) & (~df_gps["rootCategory"].isin(df_checkin["rootCategory"].unique()))]
    df_label = pd.concat([df_checkin, df_gps_label], ignore_index=True)

    catname_catind = dict(zip(df_label["rootCategory"], df_label["categoryIndex"]))
    catname_catind = {k: v for k, v in catname_catind.items() if v != -1}
    catname_catind["Non-Target"] = cfg.const.vocab.cat

    return {v: k for k, v in catname_catind.items()}


def plot(y: np.ndarray, pred: np.ndarray, cfg: DictConfig, catind_catname: Dict[int, str], out_dir: str) -> None:
    y_bin = np.where(y == cfg.const.vocab.cat, 0, 1)
    pred_bin = np.where(pred == cfg.const.vocab.cat, 0, 1)
    cat = np.unique(y)

    prec_per_cat = precision_score(y, pred, labels=cat, average=None, zero_division=0)
    rec_per_cat = recall_score(y, pred, labels=cat, average=None, zero_division=0)
    bin_prec_per_cat = precision_score(y_bin, pred_bin, labels=[1, 0], average=None, zero_division=0)
    bin_rec_per_cat = recall_score(y_bin, pred_bin, labels=[1, 0], average=None, zero_division=0)

    plot_per_category_scores(prec_per_cat, "precision", y, catind_catname, out_dir)
    plot_per_category_scores(rec_per_cat, "recall", y, catind_catname, out_dir)
    plot_per_bin_category_scores(bin_prec_per_cat, "precision", out_dir)
    plot_per_bin_category_scores(bin_rec_per_cat, "recall", out_dir)
    plot_per_category_hist(pred, y, cfg.const.vocab.cat, catind_catname, out_dir)


@hydra.main(version_base="1.1", config_path="../conf", config_name="main")
def main(cfg: DictConfig) -> None:
    init_lib(cfg)

    out_dir = os.path.join(
        cfg.path.folder.result, cfg.const.unlabel_cat, str(cfg.const.sample.frac), cfg.train.loss.type
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with mlflow.start_run(run_name=cfg.train.loss.type):
        mlflow.log_param("alpha", cfg.train.loss.alpha)
        mlflow.log_param("unlabel_cat", cfg.const.unlabel_cat)
        mlflow.log_param("frac", cfg.const.sample.frac)
        fold_scores: List[Dict[str, float]] = []

        for fold in range(cfg.const.folds):
            loader_dir = f"{cfg.path.folder.loader}/{fold}"
            with open(f"{loader_dir}/loader.txt", "rb") as f:
                loader_dict = pickle.load(f)
            with open(f"{loader_dir}/priors.pk", "rb") as f:
                priors = pickle.load(f)
            loader_train = loader_dict["loader_train"]
            loader_test = loader_dict["loader_test"]

            model = Model(cfg, DEVICE).to(DEVICE)
            opt = optim.Adam(model.parameters(), cfg.train.opt.lr)
            loss_fn = LossHandler(priors, cfg, DEVICE)

            score_names = ["Recall_All", "Recall_Tar", "Precision_Tar", "Recall_Non", "Precision_Non"]
            print("\t\t\t" + "\t".join(score_names))

            for epoch in range(cfg.train.param.num_epochs):
                train(model, loader_train, loss_fn, opt)
                y_test, pred_test = test(model, loader_test, cfg)
                scores = evaluate(y_test, pred_test, cfg)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{cfg.train.param.num_epochs}", end="\t")
                print("\t\t".join(f"{score:.4f}" for score in scores))

            last_metrics = dict(zip(score_names, scores))
            fold_scores.append(last_metrics)

            if fold == 0:  # Plot the first result
                mlflow.log_param("num_epochs", epoch + 1)
                catind_catname = get_catind_catname(cfg)
                plot(y_test, pred_test, cfg, catind_catname, out_dir)

        metrics = {key: np.mean([fold[key] for fold in fold_scores]) for key in fold_scores[0].keys()}
        metrics.update(
            {f"std_{key}": np.std([fold[key] for fold in fold_scores], ddof=1) for key in fold_scores[0].keys()}
        )
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))


if __name__ == "__main__":
    main()
