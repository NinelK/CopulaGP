import torch
from torch import Tensor
from typing import Any
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform
from torch.distributions.transformed_distribution import TransformedDistribution #for Flow

from .distributions import IndependenceCopula, GaussianCopula, FrankCopula, ClaytonCopula, GumbelCopula, StudentTCopula, MixtureCopula
from .dist_transform import NormTransform
from .models import Mixed_GPInferenceModel #to check input into input_information
from . import conf

class Copula_Likelihood_Base(Likelihood):
    def __init__(self, **kwargs: Any): 
        super(Likelihood, self).__init__()
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

class IndependenceCopula_Likelihood(Likelihood):
    def __init__(self, **kwargs: Any):
        super(Likelihood, self).__init__()
        self.copula = IndependenceCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Independence'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return f

    def expected_log_prob(self, target: Tensor, *params: Any, **kwargs: Any) -> Tensor:
        return torch.zeros(1)

    def forward(self, *params: Any, **kwargs: Any) -> IndependenceCopula:
        return self.copula()

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
            return (2*base_distributions.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(f) - 1)
        else:
            return (2*base_distributions.Normal(0,1).cdf(f) - 1)

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta

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
        return 0.3*f + f.sign()*(0.3*f)**2 

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta/conf.Frank_Theta_Max

class ClaytonCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = ClaytonCopula
        self.isrotatable = True
        self.rotation = rotation
        self.name = 'Clayton'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return (0.3*f+1e-4).exp()
        #maps (-inf, +inf) to [0,max]

    def normalize(self, theta: Tensor) -> Tensor:
        if (self.rotation == '90°') | (self.rotation == '270°'):
            return -theta/conf.Clayton_Theta_Max
        else:
            return theta/conf.Clayton_Theta_Max

class GumbelCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GumbelCopula
        self.isrotatable = True
        self.rotation = rotation
        self.name = 'Gumbel'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return (0.3*f+1e-4).exp() + 1.0

    def normalize(self, theta: Tensor) -> Tensor:
        if (self.rotation == '90°') | (self.rotation == '270°'):
            return -(theta-1)/(conf.Gumbel_Theta_Max-1)
        else:
            return (theta-1)/(conf.Gumbel_Theta_Max-1)

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
        self.likelihoods = likelihoods
        self.waic_samples = conf.waic_samples
        self.copula = MixtureCopula
        if theta_sharing is not None:
            self.theta_sharing = theta_sharing
        else:
            self.theta_sharing = torch.arange(0,len(likelihoods)).long()

    def WAIC(self, gp_distr: MultivariateNormal, target: Tensor, combine_terms=True):
        '''
            Computes WAIC
        Args:
            :attr:`gp_distr` (:class:`gpytorch.distributions.MultivariateNormal`)
                Trained Gaussian Process distribution
            :attr:`target` (:class:`torch.Tensor`)
                Values of :math:`y`.
            :attr:`n` (:class:`int`)
                Number of points in input space for logprob estimation
            :attr:`length` (:class:`float`)
                Length of the input space in physical units (seconds/cm/...). Provides scale for derivative in FI.
        Returns
            `FI` (Fisher information)
            `MI` (Mutual information)
        '''
        copulas = [lik.copula for lik in self.likelihoods]
        rotations = [lik.rotation for lik in self.likelihoods]

        with torch.no_grad():
            f_samples = gp_distr.rsample(torch.Size([self.waic_samples]))
            thetas, mixes = self.gplink_function(f_samples)
            log_prob = self.copula(thetas,mixes,copulas,rotations=rotations).\
                    log_prob(target).detach()
            pwaic = torch.var(log_prob,dim=0).sum()
            sum_prob = torch.exp(log_prob).sum(dim=0)
            N = torch.ones_like(pwaic)*self.waic_samples
            lpd=(sum_prob.log()-N.log()).sum() # sum_M log(1/N * sum^i_S p(y|theta_i)), where N is train_x.shape[0]
        #torch.cuda.empty_cache() 

        if combine_terms:
            return (lpd-pwaic) #=WAIC
        else:
            return lpd,pwaic

    def input_information(self, model: Mixed_GPInferenceModel, samples: Tensor, n: int, length: float, ignore_GP_uncertainty=False):
        '''
            Computes Fisher Information and Mutual information
        Args:
            :attr:`model` (:class:`bvcopula.models.Mixed_GPInferenceModel`)
                Trained Gaussian Process for copual parameter inference
            :attr:`samples` (:class:`torch.Tensor`)
                Values of :math:`y`.
            :attr:`n` (:class:`int`)
                Number of points in input space for logprob estimation
            :attr:`length` (:class:`float`)
                Length of the input space in physical units (seconds/cm/...). 
                Provides scale for derivative in FI.
            :attr:`ignore_GP_uncertainty` (:class:`bool`)
                If True, then mean value of GP is used. 
                Otherwise (False), full doubly-stochastic model is used for logprob estimation.
        Returns
            `FI` (Fisher information)
            `MI` (Mutual information)
        '''
        ds = length/n
        logprob = torch.empty([n+1,samples.shape[0]])
        if ignore_GP_uncertainty:
            points = torch.arange(n+1).float()/n

        if samples.is_cuda:
            get_cuda_device = samples.get_device()
            logprob = logprob.cuda(device=get_cuda_device)
            if ignore_GP_uncertainty:
                points = points.cuda(device=get_cuda_device)

        assert(model.likelihood.likelihoods==self.likelihoods) #check that it is actually a parent model

        copulas = [lik.copula for lik in self.likelihoods]
        rotations = [lik.rotation for lik in self.likelihoods]

        with torch.no_grad():
            if ignore_GP_uncertainty:
                thetas, mixes = self.gplink_function(model(points).mean)
                thetas = thetas.expand(samples.shape[:1]+thetas.shape)
                mixes = mixes.expand(samples.shape[:1]+mixes.shape)
                thetas = torch.einsum('ijk->jki', thetas) # now: [copulas, positions, samples]
                mixes = torch.einsum('ijk->jki', mixes)
                logprob = model.likelihood.copula(thetas,mixes,copulas,rotations=rotations).log_prob(samples)
            else:
                for i in range(n+1): #if GPU memory allows, can parallelize this cycle as well
                    points = torch.ones_like(samples[...,0])*(i/n)
                    functions = model(points)
                    log_prob_lambda = lambda function_samples: model.likelihood.forward(function_samples).log_prob(samples)
                    logprob[i] = model.likelihood.quadrature(log_prob_lambda, functions) 
                    
            #calculate FI
            FI = torch.empty_like(logprob[...,0])
            FI[0] = ((logprob[0].exp())*((logprob[1]-logprob[0])/(ds/2)).pow(2)).sum()
            FI[n] = ((logprob[n].exp())*((logprob[n]-logprob[n-1])/(ds/2)).pow(2)).sum()
            for i in range(1,n):
                FI[i] = ((logprob[i].exp())*((logprob[i+1]-logprob[i-1])/ds).pow(2)).sum()
                
            #now calculate MI    
            # P(r) = integral P(r|s) P(s) ds
            Pr = torch.zeros(samples.shape[0]).cuda(device=get_cuda_device)
            for i in range(n+1):
                Pr += logprob[i].exp().detach()*(1/(n+1))
            MIs=0
            for i in range(n+1):    
                MIs+= 1/(n+1)*logprob[i].exp()*(logprob[i]-Pr.log()) # sum p(r|s) * log p(r|s)/p(r)
            MI = MIs.sum()     
        return FI, MI                        

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
            return res[res.abs()!=float("inf")].sum()
        else: #use Gauss-Hermite quadrature
            def log_prob_lambda (function_samples):
                logprob = self.forward(function_samples).log_prob(target, safe=True)
                #print(logprob.min(),logprob.max(),logprob.mean(),logprob.std())
                return logprob
            log_prob = self.quadrature(log_prob_lambda, input)
            if weights is not None:
                log_prob *= weights
            #print(log_prob.min(),log_prob.max(),log_prob.mean())
            assert torch.all(log_prob==log_prob)
            res = log_prob[log_prob.abs()!=float("inf")].sum(tuple(range(-1, -len(input.event_shape) - 1, -1)))
            return res

    def gplink_function(self, f: Tensor, normalized_thetas=False) -> Tensor:
        """
        GP link function transforms the GP latent variable `f` into :math:`\theta`,
        which parameterizes the distribution in :attr:`forward` method as well as the
        log likelihood of this distribution defined in :attr:`expected_log_prob`.
        """
        num_copulas = len(self.likelihoods)
        num_indep_thetas = self.theta_sharing.max() + 1
        assert num_copulas + num_indep_thetas - 1==f.shape[-2] #independent thetas + mixing concentrations - 1 (dependent)

        lr_ratio = .5 # lr_mix / lr_thetas
        #.5 works well for MCMC, .25 for GH

        thetas, mix = [], []
        prob_rem = torch.ones_like(f[...,0,:]) #1-x1, x1(1-x2), x1x2(1-x3)...

        if f.is_cuda:
            get_cuda_device = f.get_device()
        
            for i, lik in enumerate(self.likelihoods):
                theta = lik.gplink_function(f[...,self.theta_sharing[i],:])
                if normalized_thetas==True:
                    theta = lik.normalize(theta)
                thetas.append(theta)
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
                theta = lik.gplink_function(f[...,self.theta_sharing[i],:])
                if normalized_thetas==True:
                    theta = lik.normalize(theta)
                thetas.append(theta)
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
