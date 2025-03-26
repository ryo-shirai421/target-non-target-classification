from typing import Dict

import torch
from omegaconf import DictConfig
from torch import Tensor, nn


class LossHandler(nn.Module):
    def __init__(self, priors: Dict[str, Dict[int, torch.Tensor]], cfg: DictConfig, device: torch.device) -> None:
        super(LossHandler, self).__init__()
        self.vocab_cat = cfg.const.vocab.cat
        self.loss_type = cfg.train.loss.type
        self.alpha = cfg.train.loss.alpha
        self.pn_loss = nn.CrossEntropyLoss()
        self.nu_loss = {cat: NULoss(prior=priors["nu_loss"][cat]) for cat in range(self.vocab_cat)}
        self.pu_loss = {cat: PULoss(prior=priors["pu_loss"][cat]) for cat in range(self.vocab_cat)}
        self.unl_ind = -1
        self.device = device

    def ce_loss(self, pred: Tensor, y: Tensor) -> Tensor:
        label_ind = torch.nonzero(y != self.unl_ind).squeeze()
        if label_ind.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        loss = self.pn_loss(pred[label_ind], y[label_ind])
        return loss.to(self.device) if isinstance(loss, Tensor) else torch.tensor(loss, device=self.device)

    def NUL(self, pred: Tensor, pred_n: Tensor, y: Tensor, cat_candi: Tensor) -> Tensor:
        ce_loss = self.ce_loss(pred, y)

        checkin_ind = torch.nonzero(y != self.unl_ind).squeeze()
        gps_ind = torch.nonzero(y == self.unl_ind).squeeze()
        pred_checkin = pred_n[checkin_ind]
        y_checkin = y[checkin_ind]
        pred_gps = pred_n[gps_ind]
        cat_candi_gps = cat_candi[gps_ind]

        nu_losses = torch.zeros(self.vocab_cat).to(self.device)
        for cat in range(self.vocab_cat):
            pred_checkin_neg = pred_checkin[torch.nonzero(y_checkin != cat).squeeze()]
            checkin_neg_proba = 1 - pred_checkin_neg[:, cat]

            pred_gps_neg = pred_gps[cat_candi_gps[:, cat] == 0]
            gps_neg_proba = 1 - pred_gps_neg[:, cat]

            neg_proba = torch.cat((checkin_neg_proba, gps_neg_proba))

            pred_gps_unl = pred_gps[cat_candi_gps[:, cat] == 1]
            unl_proba = 1 - pred_gps_unl[:, cat]

            input = torch.cat((neg_proba, unl_proba)).squeeze()
            target = torch.cat((torch.zeros(neg_proba.shape[0]), -torch.ones(unl_proba.shape[0]))).to(self.device)
            nu_losses[cat] = self.nu_loss[cat](input, target).to(self.device)

        nu_loss = torch.sum(nu_losses)
        loss = nu_loss + self.alpha * ce_loss
        return loss if isinstance(loss, Tensor) else torch.tensor(loss, device=self.device)

    def PNL(self, pred: Tensor, y: Tensor) -> Tensor:
        y = torch.where(y == self.unl_ind, self.vocab_cat, y)
        loss = self.pn_loss(pred, y)
        return loss.to(self.device) if isinstance(loss, Tensor) else torch.tensor(loss, device=self.device)

    def PUL(self, pred_n: Tensor, y: Tensor) -> Tensor:
        checkin_ind = torch.nonzero(y != self.unl_ind).flatten()
        gps_ind = torch.nonzero(y == self.unl_ind).flatten()
        pred_checkin = pred_n[checkin_ind]
        y_checkin = y[checkin_ind]
        pred_gps = pred_n[gps_ind]

        pu_losses = torch.zeros(self.vocab_cat).to(self.device)
        for cat in range(self.vocab_cat):
            pos_proba = pred_checkin[torch.nonzero(y_checkin == cat).flatten()][:, cat]
            unl_proba = pred_gps[:, cat]

            input = torch.cat((pos_proba, unl_proba)).squeeze()
            target = torch.cat((torch.ones(pos_proba.shape[0]), -torch.ones(unl_proba.shape[0]))).to(self.device)
            pu_losses[cat] = self.pu_loss[cat](input, target).to(self.device)

        loss = torch.sum(pu_losses)
        return loss if isinstance(loss, Tensor) else torch.Tensor(loss, device=self.device)

    def __call__(self, pred: Tensor, pred_n: Tensor, y: Tensor, cat_candi: Tensor) -> Tensor:
        if self.loss_type == "NUL":
            return self.NUL(pred, pred_n, y, cat_candi)
        elif self.loss_type == "PNL":
            return self.PNL(pred, y)
        elif self.loss_type == "PUL":
            return self.PUL(pred_n, y)
        else:
            raise ValueError("No matching loss.")


class NULoss(nn.Module):
    def __init__(self, prior: Tensor) -> None:
        super(NULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.negative = 0
        self.unlabeled = -1
        self.min_count = 1

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        assert inp.shape == target.shape
        if inp.is_cuda:
            self.prior = self.prior.cuda()

        negative, unlabeled = target == self.negative, target == self.unlabeled
        negative, unlabeled = negative.type(torch.float), unlabeled.type(torch.float)

        n_negative, n_unlabeled = (
            torch.clamp(torch.sum(negative), min=self.min_count),
            torch.clamp(torch.sum(unlabeled), min=self.min_count),
        )

        y_negative = (1 - inp) * negative
        y_negative_inv = inp * negative
        y_unlabeled = inp * unlabeled

        negative_risk = self.prior * torch.sum(y_negative) / n_negative
        positive_risk = -self.prior * torch.sum(y_negative_inv) / n_negative + torch.sum(y_unlabeled) / n_unlabeled

        if positive_risk < 0:
            return negative_risk

        return positive_risk + negative_risk


class PULoss(nn.Module):
    def __init__(self, prior: Tensor) -> None:
        super(PULoss, self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.positive = 1
        self.unlabeled = -1
        self.min_count = 1

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        assert inp.shape == target.shape
        if inp.is_cuda:
            self.prior = self.prior.cuda()

        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)

        n_positive, n_unlabeled = (
            torch.clamp(torch.sum(positive), min=self.min_count),
            torch.clamp(torch.sum(unlabeled), min=self.min_count),
        )

        y_positive = (1 - inp) * positive
        y_positive_inv = inp * positive
        y_unlabeled = inp * unlabeled

        positive_risk = self.prior * torch.sum(y_positive) / n_positive
        negative_risk = -self.prior * torch.sum(y_positive_inv) / n_positive + torch.sum(y_unlabeled) / n_unlabeled

        if negative_risk < 0:
            return positive_risk

        return positive_risk + negative_risk
