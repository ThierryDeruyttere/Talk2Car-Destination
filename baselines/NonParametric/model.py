import os
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import pytorch_lightning as pl
from torch.nn import init
from talk2car import Talk2Car_Detector
from torch.utils.data import DataLoader
import mmfp_utils
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from resnet import resnet
from flownet import FlowNetS

class NonParametric(pl.LightningModule):
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
        parser.add_argument("--unrolled", action="store_true")
        parser.add_argument("--use_ref_obj", action="store_true")
        parser.add_argument("--width", type=int, default=1200)
        parser.add_argument("--height", type=int, default=800)
        parser.add_argument("--encoder", type=str, default="ResNet-18")
        parser.add_argument("--threshold", type=float, default=1)
        parser.add_argument("--gaussian_size", type=int, default=11)
        parser.add_argument("--gaussian_sigma", type=int, default=3)

        return parser

    def __init__(self, hparams):
        super(NonParametric, self).__init__()
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

        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim + 768, 1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(),
        )
        self.x_prob = nn.Linear(512, self.input_width)
        self.y_prob = nn.Linear(512, self.input_height)

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
        img_encoding = self.encoder(x)
        hidden = self.regressor(torch.cat([img_encoding, command_embedding], dim=-1))
        x_prob = self.x_prob(hidden)
        y_prob = self.y_prob(hidden)

        return x_prob, y_prob


    def training_step(self, batch, batch_idx):
        x, (gt_x_axis, gt_y_axis) = batch["x"].float(), batch["y"]
        gt_end_pos = batch["end_pos"]
        [B, N, _] = gt_end_pos.shape
        command_embedding = batch["command_embedding"]

        x_probs, y_probs = self.forward(x, command_embedding)

        loss = F.kl_div(
            torch.log_softmax(x_probs, dim=-1), gt_x_axis, reduction="batchmean"
        ) + F.kl_div(
            torch.log_softmax(y_probs, dim=-1), gt_y_axis, reduction="batchmean"
        )
        x_probs = F.softmax(x_probs, dim=1)
        y_probs = F.softmax(y_probs, dim=1)

        gt_x = gt_end_pos[:, :, 0].long()
        gt_y = gt_end_pos[:, :, 1].long()
        gt_coord = torch.cat((gt_x.unsqueeze(2), gt_y.unsqueeze(2)), dim=2)
        prob_grid = x_probs.unsqueeze(2).bmm(y_probs.unsqueeze(1))
        log_prob_grid = torch.log(prob_grid)
        log_py = torch.gather(
            log_prob_grid.unsqueeze(1).repeat(1, N, 1, 1).view(B, N, -1),
            2,
            (gt_coord[:, :, 0] * log_prob_grid.shape[-1] + gt_coord[:, :, 1]).unsqueeze(2)
        ).squeeze().mean(dim=-1)
        nll = -1.0 * log_py.mean()

        x_distr = D.Categorical(probs=x_probs)
        y_distr = D.Categorical(probs=y_probs)
        pred_coord = torch.cat(
            (
                x_distr.sample(sample_shape=(1000,)).unsqueeze(2),
                y_distr.sample(sample_shape=(1000,)).unsqueeze(2),
            ),
            dim=2,
        ).float()
        pred_coord = pred_coord.permute(1, 0, 2)
        img_input_size = torch.tensor([self.input_width, self.input_height]).to(pred_coord)

        # Take smallest distance to one of the three end points
        distances = (
            (gt_end_pos.unsqueeze(1) - pred_coord.unsqueeze(2))
        )
        distances = distances / img_input_size * self.to_meters.to(distances)
        distances = distances.norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()

        ade = distances.mean(dim=-1)
        ade = ade.mean()

        demd = 0.0
        pred_coord = (pred_coord / img_input_size).cpu().numpy()
        for i in range(B):
            demd += mmfp_utils.wemd_from_pred_samples(pred_coord[i])
        demd /= B

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_nll", nll, on_step=False, on_epoch=True)
        self.log("train_nll", nll, on_step=False, on_epoch=True)
        self.log("train_pa", pa, on_step=False, on_epoch=True)
        self.log("train_ade", ade, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, (gt_x_axis, gt_y_axis) = batch["x"].float(), batch["y"]
        gt_end_pos = batch["end_pos"]
        [B, N, _] = gt_end_pos.shape
        command_embedding = batch["command_embedding"]

        x_probs, y_probs = self.forward(x, command_embedding)

        loss = F.kl_div(
            torch.log_softmax(x_probs, dim=-1), gt_x_axis, reduction="batchmean"
        ) + F.kl_div(
            torch.log_softmax(y_probs, dim=-1), gt_y_axis, reduction="batchmean"
        )
        x_probs = F.softmax(x_probs, dim=1)
        y_probs = F.softmax(y_probs, dim=1)

        gt_x = gt_end_pos[:, :, 0].long()
        gt_y = gt_end_pos[:, :, 1].long()
        gt_coord = torch.cat((gt_x.unsqueeze(2), gt_y.unsqueeze(2)), dim=2)
        prob_grid = x_probs.unsqueeze(2).bmm(y_probs.unsqueeze(1))
        log_prob_grid = torch.log(prob_grid)
        log_py = torch.gather(
            log_prob_grid.unsqueeze(1).repeat(1, N, 1, 1).view(B, N, -1),
            2,
            (gt_coord[:, :, 0] * log_prob_grid.shape[-1] + gt_coord[:, :, 1]).unsqueeze(2)
        ).squeeze().mean(dim=-1).mean()
        nll = -1.0 * log_py

        x_distr = D.Categorical(probs=x_probs)
        y_distr = D.Categorical(probs=y_probs)
        pred_coord = torch.cat(
            (
                x_distr.sample(sample_shape=(1000,)).unsqueeze(2),
                y_distr.sample(sample_shape=(1000,)).unsqueeze(2),
            ),
            dim=2,
        ).float()
        pred_coord = pred_coord.permute(1, 0, 2)
        img_input_size = torch.tensor([self.input_width, self.input_height]).to(pred_coord)

        # Take smallest distance to one of the three end points
        distances = (
                (gt_end_pos.unsqueeze(1) - pred_coord.unsqueeze(2))
        )
        distances = distances / img_input_size * self.to_meters.to(distances)
        distances = distances.norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()

        ade = distances.mean(dim=-1)
        ade = ade.mean()

        demd = 0.0
        pred_coord = (pred_coord / img_input_size).cpu().numpy()
        for i in range(B):
            demd += mmfp_utils.wemd_from_pred_samples(pred_coord[i])
        demd /= B

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_nll", nll, on_step=False, on_epoch=True)
        self.log("val_pa", pa, on_step=False, on_epoch=True)
        self.log("val_ade", ade, on_step=False, on_epoch=True)
        self.log("val_demd", demd, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, (gt_x_axis, gt_y_axis) = batch["x"].float(), batch["y"]
        gt_end_pos = batch["end_pos"]
        [B, N, _] = gt_end_pos.shape
        command_embedding = batch["command_embedding"]

        x_probs, y_probs = self.forward(x, command_embedding)

        loss = F.kl_div(
            torch.log_softmax(x_probs, dim=-1), gt_x_axis, reduction="batchmean"
        ) + F.kl_div(
            torch.log_softmax(y_probs, dim=-1), gt_y_axis, reduction="batchmean"
        )
        x_probs = F.softmax(x_probs, dim=1)
        y_probs = F.softmax(y_probs, dim=1)

        gt_x = gt_end_pos[:, :, 0].long()
        gt_y = gt_end_pos[:, :, 1].long()
        gt_coord = torch.cat((gt_x.unsqueeze(2), gt_y.unsqueeze(2)), dim=2)
        prob_grid = x_probs.unsqueeze(2).bmm(y_probs.unsqueeze(1))
        log_prob_grid = torch.log(prob_grid)
        log_py = torch.gather(
            log_prob_grid.unsqueeze(1).repeat(1, N, 1, 1).view(B, N, -1),
            2,
            (gt_coord[:, :, 0] * log_prob_grid.shape[-1] + gt_coord[:, :, 1]).unsqueeze(2)
        ).squeeze().mean(dim=-1).mean()
        nll = -1.0 * log_py

        x_distr = D.Categorical(probs=x_probs)
        y_distr = D.Categorical(probs=y_probs)
        pred_coord = torch.cat(
            (
                x_distr.sample(sample_shape=(1000,)).unsqueeze(2),
                y_distr.sample(sample_shape=(1000,)).unsqueeze(2),
            ),
            dim=2,
        ).float()
        pred_coord = pred_coord.permute(1, 0, 2)
        img_input_size = torch.tensor([self.input_width, self.input_height]).to(pred_coord)

        # Take smallest distance to one of the three end points
        distances = (
            (gt_end_pos.unsqueeze(1) - pred_coord.unsqueeze(2))
        )
        distances = distances / img_input_size * self.to_meters.to(distances)
        distances = distances.norm(2, dim=-1).min(-1)[0]

        corrects = distances < self.hparams.threshold
        pa = corrects.sum(dim=-1) / corrects.shape[1]
        pa = pa.mean()

        ade = distances.mean(dim=-1)
        ade = ade.mean()

        demd = 0.0
        pred_coord = (pred_coord / img_input_size).cpu().numpy()
        for i in range(B):
            demd += mmfp_utils.wemd_from_pred_samples(pred_coord[i])
        demd /= B

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_nll", nll, on_step=False, on_epoch=True)
        self.log("test_pa", pa, on_step=False, on_epoch=True)
        self.log("test_ade", ade, on_step=False, on_epoch=True)
        self.log("test_demd", demd, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "nll": nll,
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
            use_ref_obj=self.use_ref_obj,
            gaussian_size=self.hparams.gaussian_size,
            gaussian_sigma=self.hparams.gaussian_sigma
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
