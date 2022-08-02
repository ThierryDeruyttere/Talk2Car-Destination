"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
import math

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)

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
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        # sigma = torch.exp(self.sigma(minibatch))
        sigma = 1 + F.elu(self.sigma(minibatch)) + 1e-20
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

    def sample(self, pi, sigma, mu, N=1000):
        comp = D.Independent(D.Normal(loc=mu, scale=sigma), 1)
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)

        y_pred = mix.sample(sample_shape=(N,))
        return y_pred

    def compute_nll(self, pi, sigma, mu, y):
        comp = D.Independent(D.Normal(loc=mu, scale=sigma), 1)
        mix = D.MixtureSameFamily(D.Categorical(pi), comp)
        return -mix.log_prob(y)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    [B, N, _] = target.shape

    target = target.view(B*N, -1)
    pi = pi.repeat_interleave(N, dim=0)
    mu = mu.repeat_interleave(N, dim=0)
    sigma = sigma.repeat_interleave(N, dim=0)

    comp = D.Independent(D.Normal(loc=mu, scale=sigma), 1)
    mix = D.MixtureSameFamily(D.Categorical(pi), comp)
    probs = mix.log_prob(target)
    probs = probs.view(B, N)
    loss = -probs.sum(dim=-1)
    return loss
