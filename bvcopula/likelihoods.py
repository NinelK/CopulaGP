import torch
from torch import Tensor
from typing import Any
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.likelihoods.noise_models import HomoskedasticNoise, Noise
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform
from torch.distributions.transformed_distribution import TransformedDistribution #for Flow

from .distributions import GaussianCopula, FrankCopula, ClaytonCopula, GumbelCopula, StudentTCopula, MixtureCopula
from .dist_transform import NormTransform

class Copula_Likelihood_Base(Likelihood):
    def __init__(self, **kwargs: Any): #noise_covar: Noise,
        super(Likelihood, self).__init__()
        self._max_plate_nesting = 1
        self.rotation = None
        self.isrotatable = False
        self.particles = torch.Size([100])
        self.noise_covar = HomoskedasticNoise(
            noise_prior=None,
            noise_constraint=GreaterThan(1e-4),
            batch_shape=torch.Size([]),
        )

    @property
    def noise(self) -> Tensor:
        print('noise')
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        print('setter_all')
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        print('property')
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        print('setter')
        self.noise_covar.initialize(raw_noise=value)

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        #called during training
        assert torch.all(input.mean==input.mean)
        noise = self.noise_covar.noise
        if noise.is_cuda:
            get_cuda_device = noise.get_device()
            thetas = self.gplink_function(input.rsample(self.particles)+
                noise*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).sample(self.particles))
        else:
            thetas = self.gplink_function(input.rsample(self.particles)+
                noise*base_distributions.Normal(0,1).sample(self.particles))
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
        noise = self.noise_covar.noise
        if noise.is_cuda:
            get_cuda_device = noise.get_device()
            noise_samples = noise*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).sample(function_samples.shape)
        else:
            noise_samples = noise*base_distributions.Normal(0,1).sample(function_samples.shape)
        noise_samples = noise_samples.reshape(function_samples.shape)
        scale = self.gplink_function(function_samples + noise_samples)
        return self.copula(scale, rotation=self.rotation)

class GaussianCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GaussianCopula
        self.rotation = None
        self.isrotatable = False
        self.particles = torch.Size([100])
        self.name = 'Gaussian'
        self.noise_covar = HomoskedasticNoise(
            noise_prior=None,
            noise_constraint=None,
            batch_shape=torch.Size([]),
        )

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
        self.particles = torch.Size([100])
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
        self.particles = torch.Size([100])
        self.name = 'Clayton'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.sigmoid(f)*9.9#/torch.exp(torch.tensor(1.)) 
        #maps (-inf, +inf) to [0,9.9]

class GumbelCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GumbelCopula
        self.isrotatable = True
        self.rotation = rotation
        self.particles = torch.Size([100])
        self.name = 'Gumbel'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.sigmoid(f)*11.2 + 1.0
        #11. is maximum that does not crash on fully dependent samples

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
        
    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
        function_samples = input.rsample(self.particles)
        thetas, mix = self.gplink_function(function_samples)
        copula = MixtureCopula(thetas, 
                               mix, 
                               [lik.copula for lik in self.likelihoods], 
                               rotations=[lik.rotation for lik in self.likelihoods],
                               theta_sharing = self.theta_sharing)
        res = copula.log_prob(target).mean(0)
        assert res.dim()==1
        return res.sum()

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
