import torch
from torch import nn
import torch.distributions as D


class MDNLoss(nn.Module):
    def __init__(
        self, height, width
    ):
        super().__init__()
        self.image_dim = torch.tensor([height, width])

    def forward(self, mu, sigma, pi, targets):

        comp = D.Independent(
            D.Normal(
                loc=mu,
                scale=sigma
            ), 1
        )
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)

        targets = targets.permute(1, 0, 2)
        log_probs = mix.log_prob(targets).permute(1, 0)
        loss = -log_probs.mean(dim=-1)
        return loss