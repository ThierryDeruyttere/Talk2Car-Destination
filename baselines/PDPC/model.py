import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from argparse import ArgumentParser

import torch
import torch.distributions as D
import torch.nn.functional as F
import pytorch_lightning as pl
from talk2car import Talk2Car, Talk2Car_Detector
from torch.utils.data import DataLoader

from fconv import FCONV
from loss import MDNLoss
import mmfp_utils

class PDPC(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam.")
        parser.add_argument(
            "--beta2", type=float, default=0.999, help="Beta2 for Adam."
        )
        parser.add_argument(
            "--momentum", type=float, default=0.9, help="Momentum for SGD."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0.0,
            help="Weight decay for the optimizer.",
        )
        parser.add_argument(
            "--mu_decay",
            type=float,
            default=0.0,
            help="Mu vector decay weight in the loss."
        )
        parser.add_argument(
            "--sigma_decay",
            type=float,
            default=0.0,
            help="Sigma vector decay weight in the loss."
        )
        parser.add_argument(
            "--pi_entropy",
            type=float,
            default=0.0,
            help="Entropy of the pi vectors weight."
        )
        parser.add_argument("--unrolled", action="store_true")
        parser.add_argument("--use_ref_obj", action="store_true")
        parser.add_argument("--width", type=int, default=1200)
        parser.add_argument("--height", type=int, default=800)
        parser.add_argument("--n_conv", type=int, default=4)
        parser.add_argument(
            "--combine_at",
            type=int,
            default=2,
            help="After which layer in feature tower to combine with command."
        )
        parser.add_argument("--threshold", type=int, default=1)
        parser.add_argument(
            "--active_scale_inds",
            nargs="*",
            type=int,
            default=[1, 1, 1, 1]
        )
        parser.add_argument("--inner_channel", type=int, default=256)
        return parser

    def __init__(self, hparams):
        super(PDPC, self).__init__()
        self.save_hyperparameters(hparams)
        self.input_width = self.hparams.width
        self.input_height = self.hparams.height
        self.use_ref_obj = self.hparams.use_ref_obj

        if self.hparams.dataset == "Talk2Car_Detector":
            self.input_channels = 14  # 10 classes + egocar + 3 groundplan
        else:
            self.input_channels = 27  # 23 classes + egocar + 3 groundplan

        if self.use_ref_obj:
            self.input_channels += 1  # + referred

        self.predictor = FCONV(
            in_channels=self.input_channels,
            width=self.input_width,
            height=self.input_height,
            n_conv=self.hparams.n_conv,
            combine_at=self.hparams.combine_at,
            active_scale_inds=self.hparams.active_scale_inds,
            inner_channel=self.hparams.inner_channel if "inner_channel" in self.hparams else 256
        )

        self.criterion = MDNLoss(
            height=self.input_height,
            width=self.input_width
        )

        self.to_meters = torch.tensor([120.0, 80.0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def forward(self, x, command_embedding):
        mus, sigmas, pis, locations = self.predictor(x, command_embedding)
        return mus, sigmas, pis, locations

    def training_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"]
        [B ,N, _] = y.shape
        command_embedding = batch["command_embedding"]

        mu, sigma, pi, location = self.forward(x, command_embedding)
        mu, sigma, pi = self.predictor.prepare_outputs(mu, sigma, pi, location)
        pi = F.softmax(pi, dim=1)
        pi_reg = (pi * pi.log()).sum(-1)
        sigma_reg = sigma.view(B, -1).abs().sum(-1)
        # mu_reg = mu.view(B, -1).abs().sum(-1)

        loss = self.criterion(
            mu, sigma, pi, y
        )
        loss = loss.mean()\
               + self.hparams.sigma_decay * sigma_reg.mean()\
               + self.hparams.pi_entropy * pi_reg.mean()
               # + self.hparams.mu_decay * mu_reg.mean()

        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        y_pred = mix.sample(sample_shape=(1000,))
        y_pred = y_pred.permute(1, 0, 2)

        distances = (
                (y.unsqueeze(1) - y_pred.unsqueeze(2)) * self.to_meters.to(y_pred)
        ).norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()

        ade = distances.mean(dim=-1)
        ade = ade.mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_pa", pa, on_step=False, on_epoch=True)
        self.log("train_ade", ade, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"]
        [B, N, _] = y.shape
        command_embedding = batch["command_embedding"]

        mu, sigma, pi, location = self.forward(x, command_embedding)
        mu, sigma, pi = self.predictor.prepare_outputs(mu, sigma, pi, location)
        pi = F.softmax(pi, dim=1)
        pi_reg = (pi * pi.log()).sum(-1)
        sigma_reg = sigma.view(B, -1).abs().sum(-1)

        loss = self.criterion(
            mu, sigma, pi, y
        )
        loss = loss.mean() \
               + self.hparams.sigma_decay * sigma_reg.mean() \
               + self.hparams.pi_entropy * pi_reg.mean()

        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        y_pred = mix.sample(sample_shape=(1000,))
        y_pred = y_pred.permute(1, 0, 2)

        distances = (
                (y.unsqueeze(1) - y_pred.unsqueeze(2)) * self.to_meters.to(y_pred)
        ).norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()

        ade = distances.mean(dim=-1)
        ade = ade.mean()

        demd = 0.0
        for i in range(B):
            demd += mmfp_utils.wemd_from_pred_samples(y_pred[i].cpu().numpy())
        demd /= B

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_pa", pa, on_step=False, on_epoch=True)
        self.log("val_ade", ade, on_step=False, on_epoch=True)
        self.log("val_demd", demd, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"]
        [B, N, _] = y.shape
        command_embedding = batch["command_embedding"]

        mu, sigma, pi, location = self.forward(x, command_embedding)
        mu, sigma, pi = self.predictor.prepare_outputs(mu, sigma, pi, location)
        pi = F.softmax(pi, dim=1)
        pi_reg = (pi * pi.log()).sum(-1)
        sigma_reg = sigma.view(B, -1).abs().sum(-1)

        loss = self.criterion(
            mu, sigma, pi, y
        )
        loss = loss.mean() \
               + self.hparams.sigma_decay * sigma_reg.mean() \
               + self.hparams.pi_entropy * pi_reg.mean()

        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        y_pred = mix.sample(sample_shape=(1000,))
        y_pred = y_pred.permute(1, 0, 2)

        distances = (
                (y.unsqueeze(1) - y_pred.unsqueeze(2)) * self.to_meters.to(y_pred)
        ).norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()

        ade = distances.mean(dim=-1)
        ade = ade.mean()

        demd = 0.0
        for i in range(B):
            demd += mmfp_utils.wemd_from_pred_samples(y_pred[i].cpu().numpy())
        demd /= B

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_pa", pa, on_step=False, on_epoch=True)
        self.log("test_ade", ade, on_step=False, on_epoch=True)
        self.log("test_demd", demd, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "pa": pa,
            "ade": ade,
            "demd": demd
        }

    def _get_dataloader(self, split, test_id=0):
        if self.hparams.dataset == "Talk2Car_Detector":
            return Talk2Car_Detector(
                split=split,
                dataset_root=self.hparams.data_dir,
                height=self.input_height,
                width=self.input_width,
                unrolled=self.hparams.unrolled,
                use_ref_obj=self.use_ref_obj
            )
        else:
            return Talk2Car(
                split=split,
                root=self.hparams.data_dir,
                height=self.input_height,
                width=self.input_width,
                unrolled=self.hparams.unrolled,
                use_ref_obj=self.use_ref_obj
            )

    def train_dataloader(self):
        dataset = self._get_dataloader("train")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = self._get_dataloader("val")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = self._get_dataloader("test")
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
