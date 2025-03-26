import os
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def convert_surrounding_categories(row: str) -> List[int]:
    # "2 6" -> [2, 6]
    if type(row) is float:
        return []
    return list({int(elem) - 1 for elem in row.split()})


def format_checkin(df: pd.DataFrame) -> pd.DataFrame:
    df["userId"] = df["userId"] - 1
    df["surroundingCategories"] = df["surroundingCategories"].apply(convert_surrounding_categories)
    df["utcTimestamp"] = pd.to_datetime(df["utcTimestamp"])
    df["hour"] = df["utcTimestamp"].dt.hour
    df["categoryIndex"] = df["categoryIndex"] - 1
    df["label"] = 1
    return df


def make_masked_ind(catname_catind: pd.DataFrame, unlabel_cat: str) -> Dict[int, int]:
    # Mask unlabel_category with the last category index
    cat_ind = catname_catind["categoryIndex"].values
    cat_ind = np.array(sorted(cat_ind))
    unlabel_cat_ind = catname_catind[catname_catind["rootCategory"] == unlabel_cat]["categoryIndex"].values
    cat_ind_wo_unlabel = np.delete(cat_ind, unlabel_cat_ind)

    oldind_newind_map = {cat: ind for ind, cat in enumerate(cat_ind_wo_unlabel)}
    for ci in unlabel_cat_ind:
        oldind_newind_map[ci] = len(cat_ind_wo_unlabel)
    return oldind_newind_map


def update_feature_elem(row: List[int], oldind_newind_map: Dict[int, int]) -> List[int]:
    unlabel_ind = max(oldind_newind_map.values())
    return [oldind_newind_map[x] for x in row if oldind_newind_map[x] != unlabel_ind]


def make_region_vec(row: List[int], vocab_cat: int) -> List[float]:
    arr = [0.0] * vocab_cat
    for cat in row:
        arr[cat] += 1
    arr_sum = sum(arr)
    if arr_sum != 0:
        arr = [x / arr_sum for x in arr]
    return arr


def make_onehot_vec(row: List[int], vocab_cat: int) -> List[int]:
    arr = [0] * vocab_cat
    for cat in row:
        arr[cat] = 1
    return arr


def make_feature(df: pd.DataFrame, vocab_cat: int) -> pd.DataFrame:
    df["regionVector"] = df["surroundingCategories"].apply(lambda x: make_region_vec(x, vocab_cat))
    df["surroundingCategories"] = df["surroundingCategories"].apply(lambda x: make_onehot_vec(x, vocab_cat))
    return df


@hydra.main(version_base="1.1", config_path="../conf", config_name="process_csv")
def main(cfg: DictConfig) -> None:
    df_checkin = pd.read_csv(cfg.path.file.data)
    df_checkin = format_checkin(df_checkin)

    # Handle the unlabel category
    catname_catind = df_checkin[["rootCategory", "categoryIndex"]].drop_duplicates()
    oldind_newind_map = make_masked_ind(catname_catind, cfg.const.unlabel_cat)

    df_checkin["categoryIndex"] = df_checkin["categoryIndex"].map(oldind_newind_map)
    df_checkin["surroundingCategories"] = df_checkin["surroundingCategories"].apply(
        lambda row: update_feature_elem(row, oldind_newind_map)
    )
    df_checkin["evalLabel"] = df_checkin["categoryIndex"]

    # Move unlabel checkin to gps
    unlabel_ind = max(oldind_newind_map.values())
    df_gps = df_checkin[df_checkin["categoryIndex"] == unlabel_ind].copy()
    df_checkin = df_checkin[df_checkin["categoryIndex"] != unlabel_ind]

    # Randomly Move Checkin to GPS
    df_checkin_all = df_checkin.copy()
    df_checkin = df_checkin_all.sample(frac=1 - cfg.const.sample.frac, random_state=cfg.const.seed)
    df_gps = pd.concat([df_gps, df_checkin_all.drop(df_checkin.index)], ignore_index=True)
    df_gps["categoryIndex"] = -1

    vocab_cat = max(df_checkin["categoryIndex"]) + 1
    df_checkin = make_feature(df_checkin, vocab_cat)
    df_gps = make_feature(df_gps, vocab_cat)

    if not os.path.exists(cfg.path.folder.formatted):
        os.makedirs(cfg.path.folder.formatted)
    df_checkin.to_csv(cfg.path.file.checkin, index=False)
    df_gps.to_csv(cfg.path.file.gps, index=False)


if __name__ == "__main__":
    main()
