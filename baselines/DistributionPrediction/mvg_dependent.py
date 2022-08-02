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
        self.tril_diag = nn.Linear(self.in_features, out_features)
        self.tril_nondiag = nn.Linear(
            self.in_features, (out_features ** 2 - out_features) // 2
        )
        self.mu = nn.Linear(self.in_features, out_features)

    def forward(self, minibatch):
        batch_size = minibatch.shape[0]
        mu = self.mu(minibatch)
        # mu = mu.view(-1, self.out_features)
        tril_diag = self.tril_diag(minibatch)
        tril_diag = 1 + F.elu(tril_diag) + 1e-20
        # tril_diag = tril_diag.view(-1, self.out_features)
        tril_nondiag = self.tril_nondiag(minibatch)
        # tril_nondiag = tril_nondiag.view(
        #     -1, (self.out_features ** 2 - self.out_features) // 2
        # )

        tril = torch.zeros(batch_size, self.out_features, self.out_features).to(minibatch)
        tril_indices = torch.tril_indices(self.out_features, self.out_features)
        tril_indices_diag = tril_indices[:, tril_indices[0, :] == tril_indices[1, :]]
        tril_indices_nondiag = tril_indices[:, tril_indices[0, :] != tril_indices[1, :]]
        tril[:, tril_indices_diag[0], tril_indices_diag[1]] = tril_diag
        tril[:, tril_indices_nondiag[0], tril_indices_nondiag[1]] = tril_nondiag
        return mu, tril


def mvg_loss(mu, scale_tril, target):
    [B, N, _] = target.shape
    mu = mu.unsqueeze(1).repeat(1, N, 1)
    scale_tril = scale_tril.unsqueeze(1).repeat(1, N, 1, 1)

    distr = D.MultivariateNormal(loc=mu, scale_tril=scale_tril)
    probs = distr.log_prob(target)  # .squeeze(1))
    if len(probs.shape) > 2:
        probs = probs.sum(dim=-1)
    # Size of probs is batch_size x num_mixtures x num_out_features
    # Size of pi is batch_size x num_mixtures
    probs = probs.view(B, N)
    loss = -probs.sum(dim=-1)  # -log_sum_exp(probs, pi, dim=1)
    return loss
