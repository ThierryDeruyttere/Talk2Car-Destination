"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


# Based on/copy from https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Linear(in_features, num_gaussians)
        self.tril_diag = nn.Linear(in_features, out_features * num_gaussians)
        self.tril_nondiag = nn.Linear(
            in_features, (out_features ** 2 - out_features) // 2 * num_gaussians
        )
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        batch_size = minibatch.shape[0]
        pi = self.pi(minibatch)
        pi = torch.softmax(pi, dim=1)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        tril_diag = self.tril_diag(minibatch)
        tril_diag = 1 + F.elu(tril_diag) + 1e-20
        tril_diag = tril_diag.view(-1, self.num_gaussians, self.out_features)
        tril_nondiag = self.tril_nondiag(minibatch)
        tril_nondiag = tril_nondiag.view(
            -1, self.num_gaussians, (self.out_features ** 2 - self.out_features) // 2
        )
        tril = torch.zeros(
            batch_size, self.num_gaussians, self.out_features, self.out_features,
        ).to(mu.device)
        tril_indices = torch.tril_indices(self.out_features, self.out_features)
        tril_indices_diag = tril_indices[:, tril_indices[0, :] == tril_indices[1, :]]
        #tril_indices_diag = tril_indices_diag.to(mu.device)
        tril_indices_nondiag = tril_indices[:, tril_indices[0, :] != tril_indices[1, :]]
        tril[:, :, tril_indices_diag[0], tril_indices_diag[1]] = tril_diag
        tril[:, :, tril_indices_nondiag[0], tril_indices_nondiag[1]] = tril_nondiag

        return pi, tril, mu

    def sample(self, pi, scale_tril, mu, N=1000):
        comp = D.MultivariateNormal(loc=mu, scale_tril=scale_tril)
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)

        y_pred = mix.sample(sample_shape=(N,))
        return y_pred

    def compute_nll(self, pi, scale_tril, mu, y):
        comp = D.MultivariateNormal(loc=mu, scale_tril=scale_tril)
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        return -mix.log_prob(y)


def mdn_loss(pi, scale_tril, mu, target):
    [B, N, _] = target.shape

    target = target.view(B*N, -1)
    pi = pi.repeat_interleave(N, dim=0)
    mu = mu.repeat_interleave(N, dim=0)
    scale_tril = scale_tril.repeat_interleave(N, dim=0)

    comp = D.MultivariateNormal(loc=mu, scale_tril=scale_tril)
    mix = D.MixtureSameFamily(D.Categorical(pi), comp)
    probs = mix.log_prob(target)  # .squeeze(1))
    # Size of probs is batch_size x num_mixtures x num_out_features
    # Size of pi is batch_size x num_mixtures
    probs = probs.view(B, N)
    loss = -probs.sum(dim=-1)   # -log_sum_exp(probs, pi, dim=1)
    return loss
