from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, cfg: DictConfig, device: torch.device) -> None:
        super(Model, self).__init__()
        self.embed_size_user = cfg.train.model.embed_size.user
        self.embed_size_cat = cfg.train.model.embed_size.cat
        self.embed_size_hour = cfg.train.model.embed_size.hour
        self.hidden_size = cfg.train.model.hidden_size
        self.num_layers = cfg.train.model.num_layers
        self.vocab_cat = cfg.const.vocab.cat
        self.vocab_user = cfg.const.vocab.user
        self.vocab_hour = cfg.const.vocab.hour
        self.device = device
        self.unl_ind = -1

        self.user_embedding = nn.Embedding(self.vocab_user, self.embed_size_user)
        self.cat_embedding = nn.Embedding(self.vocab_cat + 1, self.embed_size_cat, padding_idx=self.vocab_cat)
        self.hour_embedding = nn.Embedding(self.vocab_hour + 1, self.embed_size_hour, padding_idx=self.vocab_hour)

        lstm_input_dim = self.vocab_cat + self.embed_size_hour + self.embed_size_user + 3
        self.lstm = nn.LSTM(lstm_input_dim, self.hidden_size, batch_first=True)
        lstm_cell_input_dim = self.embed_size_cat + self.embed_size_hour + self.embed_size_user + 2 + self.hidden_size
        self.lstm_cell = nn.LSTMCell(lstm_cell_input_dim, self.hidden_size)
        self.lstm_fc = nn.Linear(self.hidden_size, self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.vocab_cat + 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        cat_seq: Tensor,
        hour_seq: Tensor,
        region_seq: Tensor,
        time_diff_seq: Tensor,
        uid: Tensor,
        gps_cnt: Tensor,
        checkin_cnt: Tensor,
        sess_len: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, max_sess_len = cat_seq.shape

        unl_mask = (cat_seq == self.unl_ind).float()  # Mask for unlabel cat
        cat_seq = torch.where(unl_mask == 1, self.vocab_cat, cat_seq)

        seq_mask = torch.arange(max_sess_len, device=self.device).expand(batch_size, max_sess_len)
        seq_mask = seq_mask < sess_len.unsqueeze(1)  # Mask for valid seq
        hour_seq = torch.where(seq_mask == 0, self.vocab_hour, hour_seq)

        cat_seq_emb = self.cat_embedding(cat_seq)
        hour_seq_emb = self.hour_embedding(hour_seq)
        time_diff_seq = time_diff_seq.unsqueeze(2)

        user_emb = self.user_embedding(uid).unsqueeze(1).repeat(1, max_sess_len, 1)
        gps_cnt = gps_cnt.unsqueeze(1).unsqueeze(2).repeat(1, max_sess_len, 1)
        checkin_cnt = checkin_cnt.unsqueeze(1).unsqueeze(2).repeat(1, max_sess_len, 1)

        lstm_feature = torch.cat((region_seq, hour_seq_emb, user_emb, gps_cnt, checkin_cnt, time_diff_seq), 2)
        packed_feature = pack_padded_sequence(lstm_feature, sess_len.cpu(), batch_first=True, enforce_sorted=False)
        lstm_output, (_, _) = self.lstm(packed_feature)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        lstm_cell_feature = torch.cat((cat_seq_emb, hour_seq_emb, user_emb, gps_cnt, checkin_cnt), 2)
        h = torch.zeros(batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(batch_size, self.hidden_size).to(self.device)
        lstm_cell_output = torch.zeros(batch_size, max_sess_len, self.hidden_size).to(self.device)

        for t in range(max(sess_len)):
            mask_t = unl_mask[:, t].unsqueeze(-1).float()
            lstm_cell_input = torch.cat((lstm_cell_feature[:, t, :], lstm_output[:, t, :]), 1)
            h_new, c_new = self.lstm_cell(lstm_cell_input, (h, c))
            h = (1 - mask_t) * h_new + mask_t * h
            c = (1 - mask_t) * c_new + mask_t * c
            lstm_cell_output[:, t, :] = h

        lstm_feature = lstm_output[torch.arange(batch_size), sess_len - 1, :]
        lstm_cell_feature = lstm_cell_output[torch.arange(batch_size), sess_len - 2, :]

        fc_input = torch.cat((lstm_feature, lstm_cell_feature), 1)

        out = self.relu1(self.fc1(fc_input))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        out_n = self.softmax(out)

        return out, out_n
