import ast
import os
import pickle
from typing import Dict, List, Tuple, Union

import hydra
import pandas as pd
import torch
import torch.utils.data as Data
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


def convert_col_type(df: pd.DataFrame) -> pd.DataFrame:
    df["surroundingCategories"] = df["surroundingCategories"].apply(ast.literal_eval)
    df["regionVector"] = df["regionVector"].apply(ast.literal_eval)
    df["utcTimestamp"] = pd.to_datetime(df["utcTimestamp"])
    return df


class LocationDataset(object):
    def __init__(self, df: pd.DataFrame, fold: int, cfg: DictConfig) -> None:
        self.df = df
        self.fold = fold
        self.vocab_cat = cfg.const.vocab.cat
        self.unlabel_cat = cfg.const.unlabel_cat
        self.seed = cfg.const.seed
        self.day_gap = cfg.dataset.param.day_gap
        self.session_min = cfg.dataset.param.session_min
        self.batch_size = cfg.dataset.param.batch_size
        self.session_max = self.day_gap * 12
        self.unlabel_ind = -1
        self.data: Dict[str, DataLoader] = {}
        cols = [
            "cat_seq",
            "y",
            "hour_seq",
            "uid",
            "sess_len",
            "region_seq",
            "cat_candi",
            "label",
            "time_diff_seq",
            "gps_cnt",
            "checkin_cnt",
        ]
        for data_type in ["train", "test"]:
            for col in cols:
                setattr(self, f"{data_type}_{col}", [])
        self.priors: Dict[str, Dict[int, torch.Tensor]] = {}

    def train_test_split(self) -> None:
        test_start_dates = [
            pd.Timestamp("2012-10-01", tz="UTC"),
            pd.Timestamp("2012-11-01", tz="UTC"),
            pd.Timestamp("2012-12-01", tz="UTC"),
            pd.Timestamp("2013-01-01", tz="UTC"),
        ]

        test_end_dates = [
            pd.Timestamp("2012-10-31", tz="UTC"),
            pd.Timestamp("2012-11-30", tz="UTC"),
            pd.Timestamp("2012-12-31", tz="UTC"),
            pd.Timestamp("2013-01-31", tz="UTC"),
        ]

        test_start = test_start_dates[self.fold]
        test_end = test_end_dates[self.fold]

        df_test = self.df[(self.df["utcTimestamp"] >= test_start) & (self.df["utcTimestamp"] <= test_end)]
        df_train = self.df[self.df["utcTimestamp"] < test_start]

        self.df_train = df_train.sort_values(by=["userId", "utcTimestamp"]).reset_index(drop=True)
        self.df_test = df_test.sort_values(by=["userId", "utcTimestamp"]).reset_index(drop=True)

    def make_feature(self) -> None:
        self.df_train["timeDiff"] = (
            self.df_train.groupby("userId")["utcTimestamp"].diff().dt.total_seconds() / 3600
        ).fillna(0)
        self.df_test["timeDiff"] = (
            self.df_test.groupby("userId")["utcTimestamp"].diff().dt.total_seconds() / 3600
        ).fillna(0)

        time_diff_scaler = MinMaxScaler()
        self.df_train["timeDiff"] = time_diff_scaler.fit_transform(self.df_train[["timeDiff"]])
        self.df_test["timeDiff"] = time_diff_scaler.transform(self.df_test[["timeDiff"]])

        gps_cnt = self.df_train[self.df_train["categoryIndex"] == self.unlabel_ind].groupby("userId").size().to_dict()
        checkin_cnt = (
            self.df_train[self.df_train["categoryIndex"] != self.unlabel_ind].groupby("userId").size().to_dict()
        )

        self.df_train["gpsCnt"] = self.df_train["userId"].map(gps_cnt).fillna(0).astype(float)
        self.df_test["gpsCnt"] = self.df_test["userId"].map(gps_cnt).fillna(0).astype(float)

        gps_cnt_scaler = MinMaxScaler()
        self.df_train["gpsCnt"] = gps_cnt_scaler.fit_transform(self.df_train[["gpsCnt"]])
        self.df_test["gpsCnt"] = gps_cnt_scaler.transform(self.df_test[["gpsCnt"]])

        self.df_train["checkinCnt"] = self.df_train["userId"].map(checkin_cnt).fillna(0).astype(float)
        self.df_test["checkinCnt"] = self.df_test["userId"].map(checkin_cnt).fillna(0).astype(float)

        checkin_cnt_scaler = MinMaxScaler()
        self.df_train["checkinCnt"] = checkin_cnt_scaler.fit_transform(self.df_train[["checkinCnt"]])
        self.df_test["checkinCnt"] = checkin_cnt_scaler.transform(self.df_test[["checkinCnt"]])

    def make_priors(self) -> None:
        # Make priors based on cat within error range.
        cat_cnt_dict = self.df_train["categoryIndex"].value_counts().to_dict()
        label_sum = sum(cnt for cat, cnt in cat_cnt_dict.items() if cat != self.unlabel_ind)
        self.priors["nu_loss"] = {cat: torch.tensor(1 - cat_cnt_dict[cat] / label_sum) for cat in range(self.vocab_cat)}
        self.priors["pu_loss"] = {cat: torch.tensor(cat_cnt_dict[cat] / label_sum) for cat in range(self.vocab_cat)}

    def get_last_n_days_data(self, row: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
        st = row["utcTimestamp"] - pd.Timedelta(days=self.day_gap)
        df_for_n_days = df[
            (df["utcTimestamp"] >= st) & (df["utcTimestamp"] <= row["utcTimestamp"]) & (df.index <= row.name)
        ]
        return df_for_n_days.sort_values(by=["userId", "utcTimestamp"])

    def make_sessions(self, df: pd.DataFrame, data_type: str) -> None:
        data_filter = {}
        for uid, df_u in df.groupby("userId"):
            df_u = df_u.sort_values(by=["userId", "utcTimestamp"]).reset_index(drop=True)

            sess_len_list = []
            sessions = {}
            for i, (_, row) in enumerate(df_u.iterrows()):
                sess = self.get_last_n_days_data(row, df_u)
                sessions[i] = sess[
                    [
                        "categoryIndex",
                        "hour",
                        "surroundingCategories",
                        "label",
                        "timeDiff",
                        "regionVector",
                        "gpsCnt",
                        "checkinCnt",
                        "evalLabel",
                    ]
                ].values.tolist()
                sess_len_list.append(len(sess))

            sess_len_list_filter: Dict[int, int] = {}
            sessions_filter: Dict[int, List[Union[int, int, List[int], int, float, List[float], int, int]]] = {}
            for sid in sessions:
                if self.session_min <= sess_len_list[sid] <= self.session_max:
                    sess_len_list_filter[len(sessions_filter)] = sess_len_list[sid]
                    sessions_filter[len(sessions_filter)] = sessions[sid]

            if len(sessions_filter) > 0:
                data_filter[uid] = {
                    "sessions": sessions_filter,
                    "session_len_list": sess_len_list_filter,
                }

        setattr(self, f"{data_type}_filter", data_filter)

    def pad_sessions(self, data_type: str, max_sess_len: int) -> None:
        pad_vec = [-1, -1, list([-1] * (self.vocab_cat)), -1, -1, list([-1] * (self.vocab_cat)), -1, -1, -1]
        data_filter = getattr(self, f"{data_type}_filter")
        for _, data_u in data_filter.items():
            for sess in data_u["sessions"].values():
                sess.extend([pad_vec] * (max_sess_len - len(sess)))

    def get_feature(self, data_type: str) -> None:
        data_filter = getattr(self, f"{data_type}_filter")
        for uid, data_u in data_filter.items():
            sessions = data_u["sessions"]
            sess_len_list = data_u["session_len_list"]

            for sid, sess in sessions.items():
                sess_len = sess_len_list[sid]
                sess_cat = [s[0] for s in sess]
                sess_hour = [s[1] for s in sess]
                sess_surrounding_categories = [s[2] for s in sess]
                sess_label = [s[3] for s in sess]
                sess_time_diff = [s[4] for s in sess]
                sess_region_vec = [s[5] for s in sess]
                sess_gps_cnt = [s[6] for s in sess]
                sess_checkin_cnt = [s[7] for s in sess]
                sess_eval_label = [s[8] for s in sess]

                if data_type == "train":
                    getattr(self, f"{data_type}_y").append(sess_cat[sess_len - 1])
                else:
                    getattr(self, f"{data_type}_y").append(sess_eval_label[sess_len - 1])
                getattr(self, f"{data_type}_cat_candi").append(sess_surrounding_categories[sess_len - 1])
                getattr(self, f"{data_type}_label").append(sess_label[sess_len - 1])
                getattr(self, f"{data_type}_uid").append(uid)
                getattr(self, f"{data_type}_gps_cnt").append(sess_gps_cnt[sess_len - 1])
                getattr(self, f"{data_type}_checkin_cnt").append(sess_checkin_cnt[sess_len - 1])
                getattr(self, f"{data_type}_sess_len").append(sess_len)

                sess_cat[sess_len - 1] = -1  # Mask target cat

                getattr(self, f"{data_type}_cat_seq").append(sess_cat[:-1])
                getattr(self, f"{data_type}_hour_seq").append(sess_hour[:-1])
                getattr(self, f"{data_type}_region_seq").append(sess_region_vec[:-1])
                getattr(self, f"{data_type}_time_diff_seq").append(sess_time_diff[:-1])

    def get_loader(self, data_type: str) -> Data.DataLoader:
        shuffle = True if data_type == "train" else False
        dataset = Data.TensorDataset(
            torch.LongTensor(getattr(self, f"{data_type}_cat_seq")),
            torch.LongTensor(getattr(self, f"{data_type}_y")),
            torch.LongTensor(getattr(self, f"{data_type}_hour_seq")),
            torch.LongTensor(getattr(self, f"{data_type}_uid")),
            torch.LongTensor(getattr(self, f"{data_type}_sess_len")),
            torch.FloatTensor(getattr(self, f"{data_type}_region_seq")),
            torch.LongTensor(getattr(self, f"{data_type}_cat_candi")),
            torch.LongTensor(getattr(self, f"{data_type}_label")),
            torch.FloatTensor(getattr(self, f"{data_type}_time_diff_seq")),
            torch.FloatTensor(getattr(self, f"{data_type}_gps_cnt")),
            torch.FloatTensor(getattr(self, f"{data_type}_checkin_cnt")),
        )
        loader = Data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)
        return loader

    def __call__(self) -> Tuple[Dict[str, Data.DataLoader], Dict[str, Dict[int, torch.Tensor]]]:
        self.train_test_split()

        self.make_feature()  # Feature extraction after data splitting
        self.make_priors()

        self.make_sessions(self.df_train, "train")
        self.make_sessions(self.df_test, "test")
        max_sess_len = max(
            [
                max(x for x in data_filter["session_len_list"].values())
                for data_filter in getattr(self, "train_filter").values()
            ]
            + [
                max(x for x in data_filter["session_len_list"].values())
                for data_filter in getattr(self, "test_filter").values()
            ]
        )
        max_sess_len += 1  # Padding for max trajectory
        self.pad_sessions("train", max_sess_len)
        self.pad_sessions("test", max_sess_len)

        self.get_feature("train")
        self.get_feature("test")

        self.data["loader_train"] = self.get_loader("train")
        self.data["loader_test"] = self.get_loader("test")
        return self.data, self.priors


@hydra.main(version_base="1.1", config_path="../conf", config_name="make_dataset")
def main(cfg: DictConfig) -> None:
    df_checkin = pd.read_csv(cfg.path.file.checkin)
    df_gps = pd.read_csv(cfg.path.file.gps)
    df_checkin = convert_col_type(df_checkin)
    df_gps = convert_col_type(df_gps)

    df = pd.concat([df_checkin, df_gps], ignore_index=True)

    for fold in tqdm(range(cfg.const.folds)):
        dataset_builder = LocationDataset(df, fold, cfg)
        dataset, priors = dataset_builder()
        out_dir = f"{cfg.path.folder.loader}/{fold}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(f"{out_dir}/loader.txt", "wb") as f:
            pickle.dump(dataset, f)
        with open(f"{out_dir}/priors.pk", "wb") as f:
            pickle.dump(priors, f)


if __name__ == "__main__":
    main()
