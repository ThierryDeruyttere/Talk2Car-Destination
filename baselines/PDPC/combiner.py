import torch
import torch.nn as nn

from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


def make_mlp(dim_list, activation_list, batch_norm=False, dropout=0):
    """
    Generates MLP network:

    Parameters
    ----------
    dim_list : list, list of number for each layer
    activation_list : list, list containing activation function for each layer
    batch_norm : boolean, use batchnorm at each layer, default: False
    dropout : float [0, 1], dropout probability applied on each layer (except last layer)

    Returns
    -------
    nn.Sequential with layers
    """
    layers = []
    index = 0
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        activation = activation_list[index]
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        elif activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        if dropout > 0 and index < len(dim_list) - 2:
            layers.append(nn.Dropout(p=dropout))
        index += 1
    return nn.Sequential(*layers)


class Combiner(nn.Module):
    def __init__(self, method, bottleneck_dim, img_enc_h, img_enc_w):
        super(Combiner, self).__init__()

        self.method = method

        if self.method == "concat_channel":
            self.com2cnn = make_mlp(
                dim_list=[768, bottleneck_dim],
                activation_list=[None],
            )
        elif self.method in ["concat_projection", "concat_projection_skip"]:
            self.combined_proj = make_mlp(
                dim_list=[768+(img_enc_h*img_enc_w), (768+(img_enc_h*img_enc_w))//2, (img_enc_h*img_enc_w)],
                activation_list=["relu", "relu"])

        elif self.method == "channel_projection":
            self.project_img_enc = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1)
            self.com2cnn = make_mlp(
                dim_list=[768, 512, 256],
                activation_list=["relu", None],
            )
    def forward(self, command_embedding, in_decoder):
        batch, c, h, w = in_decoder.size()
        # 15, 80, 120 -> 64, 10, 15
        if self.method == "concat_channel":  # Shit
            command_embedding = self.com2cnn(command_embedding)
            command_embedding = command_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
            in_decoder = torch.cat((command_embedding, in_decoder), 1)
        elif self.method == "concat_projection":
            tmp_img_enc = in_decoder.view(batch, c, -1)  # B, 64, 150
            combined_enc = torch.cat([command_embedding.unsqueeze(1).repeat(1,c,1), tmp_img_enc], dim=-1)
            combined_enc = self.combined_proj(combined_enc.view(batch*c, -1))
            in_decoder = combined_enc.view(batch, c, h, w)
        elif self.method == "concat_projection_skip":
            tmp_img_enc = in_decoder.view(batch, c, -1)
            combined_enc = torch.cat([command_embedding.unsqueeze(1).repeat(1,c,1), tmp_img_enc], dim=-1)
            combined_enc = self.combined_proj(combined_enc.view(batch*c, -1))
            in_decoder = combined_enc.view(batch, c, h, w) + in_decoder
        elif self.method == "channel_projection":
            tmp_img_enc = self.project_img_enc(in_decoder)
            proj_com = self.com2cnn(command_embedding)
            bmm_res = proj_com.unsqueeze(1).bmm(tmp_img_enc.view(batch, 256, h * w))
            sftm = torch.softmax(bmm_res, dim=-1).view(batch, 1, h, w)
            in_decoder = sftm * in_decoder
        return in_decoder


class ChannelProjectionCombiner(nn.Module):
    def __init__(
            self,
            command_dim: int = 768,
            features_channels: int = 256
    ):
        super(ChannelProjectionCombiner, self).__init__()
        self.features_projector = nn.Conv2d(
            in_channels=features_channels,
            out_channels=features_channels,
            kernel_size=1
        )
        self.command_projector = make_mlp(
            dim_list=[command_dim, 512, features_channels],
            activation_list=["relu", None],
        )

    def forward(self, features, command_embedding):
        batch, c, h, w = features.shape
        features = self.features_projector(features)
        command_embedding = self.command_projector(command_embedding)
        bmm_res = command_embedding.unsqueeze(1).bmm(features.view(batch, c, h * w))
        sftm = torch.softmax(bmm_res, dim=-1).view(batch, 1, h, w)
        features = sftm * features
        return features


class Attention(nn.Module):
    def __init__(self, command_dim, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(command_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Conv2d(dim, self.inner_dim, 2, 1, bias=False)

        self.attend = nn.Softmax(dim=-1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, patches, command):
        b, c, h, w, heads = *patches.shape, self.heads

        q = self.to_q(command)
        kv = (self.to_kv(patches).chunk(2, dim=1))
        q2 = rearrange(q, 'b (h d) -> b h d', h=heads)
        k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), kv)

        dots = einsum('b h d, b h j d -> b h j', q2, k) * self.scale
        softm = self.attend(dots)

        out = (softm.unsqueeze(-1) * v)
        out = rearrange(out, 'b h (i j) d -> b i j (h d)', h=heads, i=h, j=w)
        return self.to_out(out).permute(0, 3, 1, 2)


class AttentionCombiner(nn.Module):
    def __init__(
            self,
            command_dim: int = 768,
            features_channels: int = 256,
    ):
        super(AttentionCombiner, self).__init__()

        self.att = Attention(
            command_dim=command_dim,
            dim=features_channels
        )
        self.norm = nn.InstanceNorm2d(num_features=features_channels)

    def forward(self, features, command_embedding):
        attn_output = self.att(features, command_embedding)
        features = self.norm(features + attn_output)
        return features


class MultiHeadAttentionCombiner(nn.Module):
    def __init__(
            self,
            command_dim: int = 768,
            features_channels: int = 256,
            n_heads: int = 4
    ):
        super(MultiHeadAttentionCombiner, self).__init__()

        self.features_projector = nn.Conv2d(
            in_channels=features_channels,
            out_channels=features_channels,
            kernel_size=1
        )
        self.command_projector = nn.Sequential(
            nn.Linear(command_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_channels),
        )
        self.att = nn.MultiheadAttention(
            embed_dim=features_channels,
            num_heads=n_heads,
            batch_first=True
        )

        self.att = Attention(
        command_dim = command_dim,
        dim = features_channels

        )
        self.norm = nn.InstanceNorm2d(num_features=features_channels)

    def forward(self, features, command_embedding):
        batch, c, h, w = features.shape
        features = self.features_projector(features)
        command_embedding = self.command_projector(command_embedding)
        features = features.view(batch, c, h * w).permute(0, 2, 1)
        command_embedding = command_embedding.unsqueeze(1)
        attn_output, attn_output_weights = self.att(features, command_embedding, command_embedding)

        attn_output = attn_output.permute(0, 2, 1).view(batch, c, h, w)
        features = features.permute(0, 2, 1).view(batch, c, h, w)
        # attn_output_weights = attn_output_weights.permute(0, 2, 1).view(batch, 1,  h, w)
        features = self.norm(features + attn_output)
        # features = attn_output
        # features = features * attn_output_weights
        return features
