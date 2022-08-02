from typing import List

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

from backbone.resnet_fpn import resnet_fpn_backbone
from combiner import ChannelProjectionCombiner, MultiHeadAttentionCombiner, AttentionCombiner


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FCONVHeadComm(nn.Module):
    def __init__(self, in_channel, n_conv=4, combine_at=2, active_scale_inds=[1, 1, 1, 1]):
        super().__init__()

        distr_tower1 = []
        distr_tower2 = []
        assert combine_at in range(0, n_conv - 1), "Impossible combiner placement"

        self.combiner = ChannelProjectionCombiner(
            command_dim=768,
            features_channels=in_channel
        )

        for i in range(combine_at):
            if i == 0:
                distr_tower1.append(
                    nn.Conv2d(256, in_channel, 3, padding=1, bias=False)
                )
            else:
                distr_tower1.append(
                    nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
                )
            distr_tower1.append(nn.GroupNorm(32, in_channel))
            distr_tower1.append(nn.ReLU())

        for i in range(combine_at, n_conv):
            distr_tower2.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            distr_tower2.append(nn.GroupNorm(32, in_channel))
            distr_tower2.append(nn.ReLU())

        self.distr_tower1 = nn.Sequential(*distr_tower1)
        self.distr_tower2 = nn.Sequential(*distr_tower2)

        self.mu_pred = nn.Conv2d(in_channel, 2, 3, padding=1)
        self.sigma_pred = nn.Conv2d(in_channel, 2, 3, padding=1)
        self.pi_pred = nn.Conv2d(in_channel, 1, 3, padding=1)

        self.apply(init_conv_std)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(4)])
        self.sigma_multipliers = [1.0, 2.0, 4.0, 8.0]
        self.mu_multipliers = [1.0, 2.0, 4.0, 8.0]
        self.active_scale_inds = active_scale_inds

    def forward(self, input, command_embedding):
        mus = []
        sigmas = []
        pis = []
        for feat, sigma_scale, sigma_multiplier, mu_multiplier, active_scale_ind in zip(
                input,
                self.scales,
                self.sigma_multipliers,
                self.mu_multipliers,
                self.active_scale_inds
        ):
            if not active_scale_ind:
                mus.append(None)
                sigmas.append(None)
                pis.append(None)
            else:
                feat = self.distr_tower1(feat)
                feat = self.combiner(feat, command_embedding)
                feat = self.distr_tower2(feat)
                mus.append(self.mu_pred(feat) * mu_multiplier)
                sigmas.append(1 + F.elu(sigma_scale(self.sigma_pred(feat) * sigma_multiplier)) + 1e-5)
                pis.append(self.pi_pred(feat))

        return mus, sigmas, pis


class FCONV(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            width: int = 320,
            height: int = 224,
            n_conv: int = 4,
            combine_at: int = 2,
            active_scale_inds: List = [1, 1, 1, 1],
            inner_channel = 256
    ):
        super().__init__()

        self.in_channels = in_channels
        self.width = width
        self.height = height
        self.n_conv = n_conv
        self.combine_at = combine_at
        self.active_scale_inds = active_scale_inds

        self.fpn = resnet_fpn_backbone(
            pretrained=False,
            norm_layer=nn.BatchNorm2d,
            trainable_layers=5,
            in_channels=self.in_channels
        )

        self.head = FCONVHeadComm(
            inner_channel, self.n_conv, self.combine_at, self.active_scale_inds
        )

        self.fpn_strides = [4, 8, 16, 32]
        self.image_dim = torch.tensor([self.height, self.width]).to(torch.float)

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.apply(freeze_bn)

    def forward(self, input, command):
        features = self.fpn(input)
        features = [features[key] for key in features.keys() if key != "pool"]
        mus, sigmas, pis = self.head(features, command)
        locations = self.compute_location(features)
        return mus, sigmas, pis, locations

    def prepare_outputs(self, mus, sigmas, pis, locations):
        mu_agg = []
        sigma_agg = []
        pi_agg = []
        for i, (mu, sigma, pi, location, active_scale_ind) in enumerate(zip(mus, sigmas, pis, locations, self.active_scale_inds)):
            if not active_scale_ind:
                continue
            [B, _, H, W] = mu.shape
            location = location / self.image_dim.to(location)
            mu = location.t().unsqueeze(0).view(1, 2, H, W) + mu
            mu = mu.view(B, 2, -1).permute(0, 2, 1)
            sigma = sigma.view(B, 2, -1).permute(0, 2, 1)
            pi = pi.view(B, -1)

            mu_agg.append(mu)
            sigma_agg.append(sigma)
            pi_agg.append(pi)

        mu_agg = torch.cat(mu_agg, dim=1)
        sigma_agg = torch.cat(sigma_agg, dim=1)
        pi_agg = torch.cat(pi_agg, dim=1)

        # Flip x and y, x is along width and y is along height in our coordinate system
        mu_agg = torch.cat(
            (mu_agg[:, :, 1].unsqueeze(2), mu_agg[:, :, 0].unsqueeze(2)),
            dim=2
        )
        sigma_agg = torch.cat(
            (sigma_agg[:, :, 1].unsqueeze(2), sigma_agg[:, :, 0].unsqueeze(2)),
            dim=2
        )
        return mu_agg, sigma_agg, pi_agg

    def compute_samples(self, mus, sigmas, pis, locations, N=1000):
        mus, sigmas, pis = self.prepare_outputs(mus, sigmas, pis, locations)
        comp = D.Independent(
            D.Normal(
                loc=mus,
                scale=sigmas
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(pis), comp)
        samples = mix.sample(sample_shape=(N,))
        samples = samples.permute(1, 0, 2)
        return samples

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_x, shift_y = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2
        return location