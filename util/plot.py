import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_per_category_hist(
    pred: np.ndarray, y: np.ndarray, vocab_cat: int, catind_catname: Dict[int, str], out_dir: str
) -> None:
    cnt = np.bincount(pred.astype(int), minlength=vocab_cat + 1)
    total_cnt = sum(cnt)
    ratios = cnt / total_cnt
    catname_score = {catind_catname[key]: ratios[key] for key in catind_catname}
    unique_values, cnt = torch.unique(torch.LongTensor(y), return_counts=True)
    catindex_catnum = dict(zip(unique_values.tolist(), cnt.tolist()))

    catname_cnt = {catind_catname[key]: catindex_catnum[key] for key in catindex_catnum}
    sorted_cat = sorted(catname_cnt, key=lambda x: catname_cnt[x], reverse=True)
    sorted_val = [catname_score[cat] for cat in sorted_cat]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_cat, sorted_val, color=["orange" if cat == "Non-Target" else "blue" for cat in sorted_cat])

    plt.xlabel("Category")

    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "hist_per_category.png"))
    plt.close()


def plot_per_category_scores(
    score: np.ndarray, metric: str, y: np.ndarray, catind_catname: Dict[int, str], out_dir: str
) -> None:
    catname_score = {catind_catname[key]: score[key] for key in catind_catname}
    unique_values, cnt = torch.unique(torch.LongTensor(y), return_counts=True)
    catindex_catnum = dict(zip(unique_values.tolist(), cnt.tolist()))

    catname_cnt = {catind_catname[key]: catindex_catnum[key] for key in catindex_catnum}
    sorted_cat = sorted(catname_cnt, key=lambda x: catname_cnt[x], reverse=True)
    sorted_val = [catname_score[cat] for cat in sorted_cat]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_cat, sorted_val, color=["orange" if cat == "Non-Target" else "blue" for cat in sorted_cat])

    plt.ylim(0, 1)
    plt.xlabel("Category")
    plt.ylabel(f"{metric.capitalize()} Values")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{metric}_per_category.png"))
    plt.close()


def plot_per_bin_category_scores(bin_met_per_cat: np.ndarray, metric: str, out_dir: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.bar(["Target", "Non-Target"], bin_met_per_cat, width=0.4, color="blue")
    plt.ylim(0, 1)
    plt.xlabel("Category")
    plt.ylabel(f"{metric.capitalize()} Values")

    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"bin_{metric}_per_category.png"))
    plt.close()
