import torch
from torch import Tensor
from typing import Any
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform

from .distributions import GaussianCopula

class GaussianCopula_Likelihood(Likelihood):
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
    
    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> GaussianCopula:
        scale = self.gplink_function(function_samples)
        return GaussianCopula(scale)