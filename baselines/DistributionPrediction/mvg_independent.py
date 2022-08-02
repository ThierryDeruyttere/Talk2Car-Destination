"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import math

import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn

# Based on/copy from https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MVG(nn.Module):
    """A multivariate gaussian network layer
    The input maps to the parameters of a multivariate normal probability distribution,
    characterized by the mean and the covariance matrix.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (mu, tril) (BxO, BxOxO): B is the batch size, and O is the number of 
            output dimensions. Mu is the mean of each
            Gaussian. Tril is the Cholesky decomposition of the covariance matrix. 
    """

    def __init__(self, in_features, out_features):
        super(MVG, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(self.in_features, out_features)
        self.sigma = nn.Linear(self.in_features, out_features)

    def forward(self, minibatch):
        batch_size = minibatch.shape[0]
        mu = self.mu(minibatch)
        sigma = self.sigma(minibatch)
        mu = mu.view(-1, self.out_features)
        sigma = 1 + F.elu(sigma) + 1e-20
        sigma = sigma.view(-1, self.out_features)

        return mu, sigma


def mvg_loss(mu, sigma, target):
    [B, N, _] = target.shape
    mu = mu.unsqueeze(1).repeat(1, N, 1)
    sigma = sigma.unsqueeze(1).repeat(1, N, 1)

    distr = D.Normal(loc=mu, scale=sigma)
    probs = distr.log_prob(target)
    if len(probs.shape) > 2:
        probs = probs.sum(dim=-1)
    probs = probs.view(B, N)
    loss = -probs.sum(dim=-1)
    return loss
