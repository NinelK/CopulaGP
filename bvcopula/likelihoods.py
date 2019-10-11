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

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, weights=None, particles=torch.Size([0]), *params: Any, **kwargs: Any) -> Tensor:
        """
        Computes the expected log likelihood (used for variational inference):
        .. math::
            \mathbb{E}_{f(x)} \left[ \log p \left( y \mid f(x) \right) \right]
        Args:
            :attr:`function_dist` (:class:`gpytorch.distributions.MultivariateNormal`)
                Distribution for :math:`f(x)`.
            :attr:`observations` (:class:`torch.Tensor`)
                Values of :math:`y`.
            :attr:`kwargs`
        Returns
            `torch.Tensor` (log probability)
        """
        #called during training
        if particles > torch.Size([0]): #do MC
            assert torch.all(input.mean==input.mean)
            thetas = self.gplink_function(input.rsample(self.particles))
            assert torch.all(thetas==thetas)
            res = self.copula(thetas, rotation=self.rotation).log_prob(target).mean(0)
            if weights is not None:
                res *= weights
            assert res.dim()==1
            assert torch.all(res==res)
            return res.sum()
        else: #use Gauss-Hermite quadrature
            log_prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(target)
            log_prob = self.quadrature(log_prob_lambda, input) 
            if weights is not None:
                log_prob *= weights
            return log_prob.sum(tuple(range(-1, -len(input.event_shape) - 1, -1)))

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
        self.name = 'Gaussian'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        if f.is_cuda:
            get_cuda_device = f.get_device()
            return 0.9999*(2*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(f) - 1)
        else:
            return 0.9999*(2*base_distributions.Normal(0,1).cdf(f) - 1)

class StudentTCopula_Likelihood(Copula_Likelihood_Base):  
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = StudentTCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Student T'

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
        self.name = 'Frank'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return (torch.sigmoid(f)-0.5)*29.8 #makes derivatives bigger and allows to keep the same learning rate as for Gaussian 

class ClaytonCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = ClaytonCopula
        self.isrotatable = True
        self.rotation = rotation
        self.name = 'Clayton'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.sigmoid(f)*9.5+1e-4#/torch.exp(torch.tensor(1.)) 
        #maps (-inf, +inf) to [0,9.9]

class GumbelCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GumbelCopula
        self.isrotatable = True
        self.rotation = rotation
        self.name = 'Gumbel'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.sigmoid(f)*11.0 + 1.0
        #11. is maximum that does not crash on fully dependent samples

class GaussianCopula_Flow_Likelihood(Likelihood):
    def __init__(self, batch_shape=torch.Size(), **kwargs: Any):
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
    
class MixtureCopula_Likelihood(Likelihood):
    def __init__(self, likelihoods, theta_sharing=None, **kwargs: Any):
        super(Likelihood, self).__init__()
        self._max_plate_nesting = 1
        self.likelihoods = likelihoods
        self.particles = torch.Size([100])
        self.copula = MixtureCopula
        if theta_sharing is not None:
            self.theta_sharing = theta_sharing
        else:
            self.theta_sharing = torch.arange(0,len(likelihoods)).long()
        
    # def expected_log_prob(self, target: Tensor, input: MultivariateNormal, weights=None, *params: Any, **kwargs: Any) -> Tensor:
    #     function_samples = input.rsample(self.particles)
    #     thetas, mix = self.gplink_function(function_samples)
    #     copula = MixtureCopula(thetas, 
    #                            mix, 
    #                            [lik.copula for lik in self.likelihoods], 
    #                            rotations=[lik.rotation for lik in self.likelihoods],
    #                            theta_sharing = self.theta_sharing)
    #     res = copula.log_prob(target).mean(0)
    #     if weights is not None:
    #         res *= weights
    #     assert res.dim()==1
    #     return res.sum()

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, weights=None, particles=torch.Size([0]), *params: Any, **kwargs: Any) -> Tensor:
        """
        Computes the expected log likelihood (used for variational inference):
        .. math::
            \mathbb{E}_{f(x)} \left[ \log p \left( y \mid f(x) \right) \right]
        Args:
            :attr:`function_dist` (:class:`gpytorch.distributions.MultivariateNormal`)
                Distribution for :math:`f(x)`.
            :attr:`observations` (:class:`torch.Tensor`)
                Values of :math:`y`.
            :attr:`kwargs`
        Returns
            `torch.Tensor` (log probability)
        """
        #called during training
        if particles > torch.Size([0]): #do MC
            function_samples = input.rsample(particles)
            thetas, mix = self.gplink_function(function_samples)
            assert torch.all(thetas==thetas)
            assert torch.all(mix==mix)
            copula = MixtureCopula(thetas, 
                               mix, 
                               [lik.copula for lik in self.likelihoods], 
                               rotations=[lik.rotation for lik in self.likelihoods],
                               theta_sharing = self.theta_sharing)
            res = copula.log_prob(target).mean(0)
            if weights is not None:
                res *= weights
            assert res.dim()==1
            assert torch.all(res==res)
            return res.sum()
        else: #use Gauss-Hermite quadrature
            log_prob_lambda = lambda function_samples: self.forward(function_samples).log_prob(target)
            log_prob = self.quadrature(log_prob_lambda, input) 
            if weights is not None:
                log_prob *= weights
            res = log_prob.sum(tuple(range(-1, -len(input.event_shape) - 1, -1)))
            return res

    def gplink_function(self, f: Tensor) -> Tensor:
        """
        GP link function transforms the GP latent variable `f` into :math:`\theta`,
        which parameterizes the distribution in :attr:`forward` method as well as the
        log likelihood of this distribution defined in :attr:`expected_log_prob`.
        """
        num_copulas = len(self.likelihoods)
        num_indep_thetas = self.theta_sharing.max() + 1
        assert num_copulas + num_indep_thetas - 1==f.shape[-2] #independent thetas + mixing concentrations - 1 (dependent)

        lr_ratio = 0.5 # lr_mix / lr_thetas

        thetas, mix = [], []
        prob_rem = torch.ones_like(f[...,0,:]) #1-x1, x1(1-x2), x1x2(1-x3)...

        if f.is_cuda:
            get_cuda_device = f.get_device()
        
            for i, lik in enumerate(self.likelihoods):
                thetas.append(lik.gplink_function(f[...,self.theta_sharing[i],:]))
                prob = torch.ones_like(f[...,0,:])
                for j in range(i):
                    p0 = (num_copulas-j-1)/(num_copulas-j)*torch.ones_like(f[...,0,:]) # 3/4, 2/3, 1/2
                    f0 = base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(p0) 
                    prob = prob*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(lr_ratio*f[...,j+num_indep_thetas,:]+f0)
                if i!=(num_copulas-1):
                    p0 = (num_copulas-i-1)/(num_copulas-i)*torch.ones_like(f[...,0,:]) # 3/4, 2/3, 1/2
                    f0 = base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(p0) 
                    prob = prob*(1.0-base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(lr_ratio*f[...,i+num_indep_thetas,:]+f0))

                mix.append(prob)
        else:
            for i, lik in enumerate(self.likelihoods):
                thetas.append(lik.gplink_function(f[...,self.theta_sharing[i],:]))
                prob = torch.ones_like(f[...,0,:])

                for j in range(i):
                    p0 = (num_copulas-j-1)/(num_copulas-j)*torch.ones_like(f[...,0,:]) # 3/4, 2/3, 1/2
                    f0 = base_distributions.Normal(0,1).icdf(p0) 
                    prob = prob*base_distributions.Normal(0,1).cdf(lr_ratio*f[...,j+num_indep_thetas,:]+f0)
                if i!=(num_copulas-1):
                    p0 = (num_copulas-i-1)/(num_copulas-i)*torch.ones_like(f[...,0,:]) # 3/4, 2/3, 1/2
                    f0 = base_distributions.Normal(0,1).icdf(p0) 
                    prob = prob*(1.0-base_distributions.Normal(0,1).cdf(lr_ratio*f[...,i+num_indep_thetas,:]+f0))

                mix.append(prob)

        stack_thetas = torch.stack(thetas)
        stack_mix = torch.stack(mix)

        assert torch.all(stack_thetas==stack_thetas)
        assert torch.all(stack_mix==stack_mix)
        return stack_thetas, stack_mix

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> MixtureCopula:
        thetas, mix = self.gplink_function(function_samples)
        return MixtureCopula(thetas, 
                             mix, 
                             [lik.copula for lik in self.likelihoods], 
                             rotations=[lik.rotation for lik in self.likelihoods],
                             theta_sharing = self.theta_sharing)
