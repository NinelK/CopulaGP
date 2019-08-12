import torch
from torch import Tensor
from typing import Any
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform
from torch.distributions.transformed_distribution import TransformedDistribution #for Flow

from .distributions import GaussianCopula, FrankCopula, ClaytonCopula, GumbelCopula, StudentTCopula, MixtureCopula
from .dist_transform import NormTransform

class Copula_Likelihood_Base(Likelihood):
    def __init__(self, **kwargs: Any):
        super(Likelihood, self).__init__()
        self._max_plate_nesting = 1
        self.rotation = None
        self.isrotatable = False
        self.particles = torch.Size([100])

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        # samples = input.rsample(self.particles)
        # thetas = self.gplink_function(samples)
        # assert torch.all(thetas!=thetas)
        # res = self.copula(thetas, rotation=self.rotation).log_prob(target).mean(0)
        # assert res.dim()==1
        # return res.sum()
        assert torch.all(input.mean==input.mean)
        thetas = self.gplink_function(input.rsample(self.particles))
        assert torch.all(thetas==thetas)
        res = self.copula(thetas, rotation=self.rotation).log_prob(target).mean(0)
        assert res.dim()==1
        assert torch.all(res==res)
        return res.sum()

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        """
        GP link function transforms the GP latent variable `f` into :math:`\theta`,
        which parameterizes the distribution in :attr:`forward` method as well as the
        log likelihood of this distribution defined in :attr:`expected_log_prob`.
        """
        pass

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> GaussianCopula:
        scale = self.gplink_function(function_samples)
        return self.copula(scale, rotation=self.rotation)

class GaussianCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GaussianCopula
        self.rotation = None
        self.isrotatable = False
        self.particles = torch.Size([100])

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        if f.is_cuda:
            get_cuda_device = f.get_device()
            return (2*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(f) - 1)
        else:
            return (2*base_distributions.Normal(0,1).cdf(f) - 1)

class StudentTCopula_Likelihood(Copula_Likelihood_Base):  
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = StudentTCopula
        self.rotation = None
        self.isrotatable = False

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        if f.is_cuda:
            get_cuda_device = f.get_device()
            return (2*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(f) - 1)
        else:
            return (2*base_distributions.Normal(0,1).cdf(f) - 1)

class FrankCopula_Likelihood(Copula_Likelihood_Base):  
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = FrankCopula
        self.rotation = None
        self.isrotatable = False
        self.particles = torch.Size([100])

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return (torch.sigmoid(2.*f)-0.5)*34.0 #makes derivatives bigger and allows to keep the same learning rate as for Gaussian 

class ClaytonCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = ClaytonCopula
        self.isrotatable = True
        self.rotation = rotation
        self.particles = torch.Size([1000])

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.clamp(torch.exp(f),0.,12.6)

class GumbelCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GumbelCopula
        self.isrotatable = True
        self.rotation = rotation
        self.particles = torch.Size([1000])

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.clamp(1.+f.exp(),1.,17.)

class MixtureCopula_Likelihood(Likelihood):
    
    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> MixtureCopula:
        mixing_vec, likelihoods = params
        likelihood = 0
        for a, lik in zip(mixing_vec, likelihoods):
            f = lik.gplink_function(function_samples)
            likelihood += lik.copula(f)*a
        return likelihood

class GaussianCopula_Flow_Likelihood(Likelihood):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), **kwargs: Any):
        batch_shape = _deprecate_kwarg_with_transform(
            kwargs, "batch_size", "batch_shape", batch_shape, lambda n: torch.Size([n])
        )
        super(Likelihood, self).__init__()
        self._max_plate_nesting = 1
    
    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        """
        GP link function transforms the GP latent variable `f` into :math:`\theta`,
        which parameterizes the distribution in :attr:`forward` method as well as the
        log likelihood of this distribution defined in :attr:`expected_log_prob`.
        """
        if f.is_cuda:
            get_cuda_device = f.get_device()
            return (2*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(f) - 1)
        else:
            return (2*base_distributions.Normal(0,1).cdf(f) - 1)
    
    @staticmethod
    def corr_mat(X: Tensor) -> Tensor:
        """
        Constructs a batch of correlation matrices with '\rho' = Tensor.
        Batch size = batch_size of `\rho`
        """
        corr_mat = torch.stack([torch.stack([torch.ones_like(X),X]),
                                torch.stack([X,torch.ones_like(X)])])
        corr_mat = torch.einsum('ij...->...ij', corr_mat)
        return corr_mat
    
    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> GaussianCopula:
        scale = self.gplink_function(function_samples)
        base_dist = base_distributions.MultivariateNormal(torch.zeros(scale.shape + torch.Size([2])),
                                                          self.corr_mat(scale))
        return TransformedDistribution(base_dist, NormTransform())