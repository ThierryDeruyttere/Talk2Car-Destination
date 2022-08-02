import os
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from argparse import ArgumentParser

import torch
from torch import nn
import torch.distributions as D
import pytorch_lightning as pl
from talk2car import Talk2Car_Detector
from torch.utils.data import DataLoader, Subset

import mmfp_utils
from resnet import resnet
from flownet import FlowNetS
import mvg_independent
import mvg_dependent

class DistributionPrediction(pl.LightningModule):
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
            "--cov_decay",
            type=float,
            default=0.0,
            help="Covariance matrix deacy weight in the loss."
        )
        parser.add_argument("--unrolled", action="store_true")
        parser.add_argument("--use_ref_obj", action="store_true")
        parser.add_argument("--width", type=int, default=1200)
        parser.add_argument("--height", type=int, default=800)
        parser.add_argument("--encoder", type=str, default="ResNet-18")
        parser.add_argument("--threshold", type=int, default=1)
        parser.add_argument("--mvg_type", type=str, default="independent")
        return parser

    def __init__(self, hparams):
        super(DistributionPrediction, self).__init__()
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

        encoder_dim = None
        if self.hparams.encoder == "FlowNet":
            encoder_dim = 1024
            self.encoder = FlowNetS(
                input_width=self.input_width,
                input_height=self.input_height,
                input_channels=self.input_channels
            )
        elif "ResNet" in self.hparams.encoder:
            if self.hparams.encoder == "ResNet":
                self.hparams.encoder = "ResNet-18"
            encoder_dim = 512
            self.encoder = resnet(
                self.hparams.encoder,
                in_channels=self.input_channels,
                num_classes=512
            )

        # self.encoder_command = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU()
        # )

        self.combiner = nn.Sequential(
            nn.Linear(encoder_dim + 768, 1024),
            # nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            # nn.Dropout(),
        )
        if self.hparams.mvg_type == "dependent":
            self.mvg_head = mvg_dependent.MVG(512, 2)
            self.criterion = mvg_dependent.mvg_loss
        elif self.hparams.mvg_type == "independent":
            self.mvg_head = mvg_independent.MVG(512, 2)
            self.criterion = mvg_independent.mvg_loss

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
        x = self.encoder(x)
        # command_embedding = self.encoder_command(command_embedding)
        x = self.combiner(torch.cat([x, command_embedding], dim=-1))
        mu, tril = self.mvg_head(x)
        return mu, tril

    def training_step(self, batch, batch_idx):
        x, y = batch["x"].float(), batch["y"]
        [B, N, _] = y.shape
        command_embedding = batch["command_embedding"]

        mu, tril_or_sigma = self.forward(x, command_embedding)
        sigma_reg = tril_or_sigma.view(B, -1).abs().sum(-1)
        loss = self.criterion(
            mu,
            tril_or_sigma,
            y
        )
        loss = loss.mean() + self.hparams.cov_decay * sigma_reg.mean()

        if self.hparams.mvg_type == "dependent":
            distr = D.MultivariateNormal(loc=mu, scale_tril=tril_or_sigma)
        else:
            distr = D.Normal(loc=mu, scale=tril_or_sigma)
        y_pred = distr.sample(sample_shape=(1000,))
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

        mu, tril_or_sigma = self.forward(x, command_embedding)
        sigma_reg = tril_or_sigma.view(B, -1).abs().sum(-1)

        loss = self.criterion(
            mu,
            tril_or_sigma,
            y
        )
        loss = loss.mean() + self.hparams.cov_decay * sigma_reg.mean()

        if self.hparams.mvg_type == "dependent":
            distr = D.MultivariateNormal(loc=mu, scale_tril=tril_or_sigma)
        else:
            distr = D.Normal(loc=mu, scale=tril_or_sigma)
        y_pred = distr.sample(sample_shape=(1000,))
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

        mu, tril_or_sigma = self.forward(x, command_embedding)
        sigma_reg = tril_or_sigma.view(B, -1).abs().sum(-1)

        loss = self.criterion(
            mu,
            tril_or_sigma,
            y
        )
        loss = loss.mean() + self.hparams.cov_decay * sigma_reg.mean()

        if self.hparams.mvg_type == "dependent":
            distr = D.MultivariateNormal(loc=mu, scale_tril=tril_or_sigma)
        else:
            distr = D.Normal(loc=mu, scale=tril_or_sigma)
        y_pred = distr.sample(sample_shape=(1000,))
        y_pred = y_pred.permute(1, 0, 2)

        distances = (
                (y.unsqueeze(1) - y_pred.unsqueeze(2)) * self.to_meters.to(y_pred)
        ).norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()
        pa = pa.item()

        ade = distances.mean(dim=-1)
        ade = ade.mean()
        ade = ade.item()

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
        return Talk2Car_Detector(
            split=split,
            dataset_root=self.hparams.data_dir,
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
