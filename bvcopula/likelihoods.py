import torch
from torch import Tensor
from typing import Any
from gpytorch.likelihoods.likelihood import Likelihood, _OneDimensionalLikelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.settings import num_likelihood_samples
from torch.distributions.transformed_distribution import TransformedDistribution #for Flow

from .distributions import IndependenceCopula, GaussianCopula, FrankCopula, ClaytonCopula, GumbelCopula, StudentTCopula, MixtureCopula
from .dist_transform import NormTransform
# from .models import MultitaskGPModel #to check input into input_information
from . import conf

class Copula_Likelihood_Base(_OneDimensionalLikelihood):
    def __init__(self): 
        super(_OneDimensionalLikelihood, self).__init__()
        self.rotation = None
        self.isrotatable = False

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        """
        GP link function transforms the GP latent variable `f` into :math:`\theta`,
        which parameterizes the distribution in :attr:`forward` method as well as the
        log likelihood of this distribution defined in :attr:`expected_log_prob`.
        """
        pass

    def forward(self, function_samples: Tensor, *params: Any) -> GaussianCopula:
        scale = self.gplink_function(function_samples)
        return self.copula(scale, rotation=self.rotation)

class IndependenceCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self):
        super(Copula_Likelihood_Base, self).__init__()
        self.copula = IndependenceCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Independence'
        assert self.name == type(self).__name__[:-17]

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return f

    def expected_log_prob(self, target: Tensor, *params: Any) -> Tensor:
        return torch.zeros(1)

    def forward(self, *params: Any) -> IndependenceCopula:
        return self.copula()

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta

class GaussianCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self):
        super(Copula_Likelihood_Base, self).__init__()
        self.copula = GaussianCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Gaussian'
        assert self.name == type(self).__name__[:-17]

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.erf(f/1.4)

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta

class StudentTCopula_Likelihood(Copula_Likelihood_Base):  
    def __init__(self):
        super(Copula_Likelihood_Base, self).__init__()
        self.copula = StudentTCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'StudentT'
        assert self.name == type(self).__name__[:-17]

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.erf(f)

class FrankCopula_Likelihood(Copula_Likelihood_Base):  
    def __init__(self):
        super(Copula_Likelihood_Base, self).__init__()
        self.copula = FrankCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Frank'
        assert self.name == type(self).__name__[:-17]

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return 0.3*f + f.sign()*(0.3*f)**2 

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta/conf.Frank_Theta_Max

class ClaytonCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, rotation=None):
        super(Copula_Likelihood_Base, self).__init__()
        self.copula = ClaytonCopula
        self.isrotatable = True
        self.rotation = rotation
        self.name = 'Clayton'
        assert self.name == type(self).__name__[:-17]

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
    def __init__(self, rotation=None):
        super(Copula_Likelihood_Base, self).__init__()
        self.copula = GumbelCopula
        self.isrotatable = True
        self.rotation = rotation
        self.name = 'Gumbel'
        assert self.name == type(self).__name__[:-17]

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return (0.3*f+1e-4).exp() + 1.0

    def normalize(self, theta: Tensor) -> Tensor:
        if (self.rotation == '90°') | (self.rotation == '270°'):
            return -(theta-1)/(conf.Gumbel_Theta_Max-1)
        else:
            return (theta-1)/(conf.Gumbel_Theta_Max-1)
    
class MixtureCopula_Likelihood(Likelihood):
    def __init__(self, likelihoods):
        super(Likelihood, self).__init__()

        # if type(likelihoods) != list:
        #     likelihoods = [likelihoods]
        assert type(likelihoods) == list
        for lik in likelihoods:
            assert type(lik).__bases__[0] == Copula_Likelihood_Base, \
                "Wrong likelihood type in the mixture"

        self.likelihoods = likelihoods

        self.copula = MixtureCopula
        self.num_copulas = len(self.likelihoods)
        self.f_size = 2*self.num_copulas - 1 # first k -- copula params, next k-1 -- mixing coefs

        # f dimensions are [f_samples dim x GP variables dim]
        # theta dimensions will be [copulas dim x f samples dim], where
        # f samples dim = batch dimension
        # note that f samples dim may be empty    

    def serialize(self):
        copula_names=[]
        for lik in self.likelihoods:
            copula_names.append([lik.name,lik.rotation]) # e.g. ['Gaussian','None']
        return copula_names

    @classmethod
    def deserialize(cls,lik_list, just_likelihoods = False):
        allowed_likelihoods = [IndependenceCopula_Likelihood,
                                GaussianCopula_Likelihood,
                                FrankCopula_Likelihood,
                                ClaytonCopula_Likelihood,
                                GumbelCopula_Likelihood]
        lookup_likelihoods = {}
        for copula_type in allowed_likelihoods:
            lookup_likelihoods[copula_type.__name__[:-17]] = copula_type
        # use the beginning of each class name as a str token for serialization

        likelihoods = []
        for lik in lik_list:
            # instantiate
            inst = lookup_likelihoods[lik[0]]()
            # set the rotation
            if inst.isrotatable:
                inst.rotation=lik[1]
            likelihoods.append(inst)
        if just_likelihoods:
            return likelihoods
        else:
            return cls(likelihoods)

    def WAIC_(self, gp_distr: MultivariateNormal, target: Tensor, combine_terms=True):
        '''
            Estimates WAIC (accuracy depends on the number of particles: conf.waic_samples)
        Args:
            :attr:`gp_distr` (:class:`gpytorch.distributions.MultivariateNormal`)
                Trained Gaussian Process distribution
            :attr:`target` (:class:`torch.Tensor`)
                Values of :math:`y`.
        Returns
            `WAIC` Widely applicable information criterion
        '''

        with torch.no_grad():
            N = target[...,0].numel() #number of data points. The last dimension is (2,) here.
            samples_shape = torch.Size([conf.waic_samples])
            f_samples = gp_distr.rsample(samples_shape) # [GP samples x GP variables x input shape]
            # from GP perspective it is [sample x batch x event] dims
            log_prob = self.get_copula(f_samples).log_prob(target).detach()
            pwaic = torch.var(log_prob,dim=0).sum()
            S = torch.ones_like(pwaic)*conf.waic_samples
            lpd=(log_prob.logsumexp(dim=0)-S.log()).sum() # sum_M log(1/N * sum^i_S p(y|theta_i)), where N is train_x.shape[0]

        if combine_terms:
            return -(lpd-pwaic)/N #=WAIC
        else:
            return lpd/N,pwaic/N

    def WAIC(self, gp_distr: MultivariateNormal, target: Tensor, combine_terms=True, waic_resamples=conf.waic_resamples):
        if combine_terms == True:
            WAIC = 0
            for rep in range(waic_resamples):
                WAIC += self.WAIC_(gp_distr=gp_distr, target=target, combine_terms=True)/waic_resamples
            return WAIC.cpu().item()
        else:
            return self.WAIC_(gp_distr=gp_distr, target=target, combine_terms=False).cpu().item()

    def get_copula(self, f):
        '''
        Returns a copula given the GP sample
        '''
        thetas, mixes = self.gplink_function(f)
        copulas = [lik.copula for lik in self.likelihoods]
        rotations = [lik.rotation for lik in self.likelihoods]
        return self.copula(thetas,mixes,copulas,rotations=rotations)

    def gplink_function(self, f: Tensor, normalized_thetas=False) -> Tensor:
        """
        GP link function transforms the GP latent variable `f` into :math:`\theta`,
        which parameterizes the distribution in :attr:`forward` method as well as the
        log likelihood of this distribution defined in :attr:`expected_log_prob`.
        """

        assert self.f_size==f.shape[-1] # = independent thetas + mixing concentrations - 1 (dependent)
        # we assume that there is 1 GP to parameterise each copula in this class

        thetas, mix = [], []
        prob_rem = torch.ones_like(f[...,0]) #1-x1, x1(1-x2), x1x2(1-x3)...
        
        for i, lik in enumerate(self.likelihoods):
            theta = lik.gplink_function(f[...,i])
            if normalized_thetas==True:
                theta = lik.normalize(theta)
            thetas.append(theta)
            prob = torch.ones_like(f[...,0])
            for j in range(i):
                p0 = (self.num_copulas-j-1)/(self.num_copulas-j)*torch.ones_like(f[...,0]) # 3/4, 2/3, 1/2
                f0 = base_distributions.Normal(torch.zeros(1, device=f.device),
                    torch.ones(1, device=f.device)).icdf(p0) 
                prob = prob*base_distributions.Normal(torch.zeros(1, device=f.device),
                    torch.ones(1, device=f.device)).cdf(conf.mix_lr_ratio*f[...,j+self.num_copulas]+f0)
            if i!=(self.num_copulas-1):
                p0 = (self.num_copulas-i-1)/(self.num_copulas-i)*torch.ones_like(f[...,0]) # 3/4, 2/3, 1/2
                f0 = base_distributions.Normal(torch.zeros(1, device=f.device),
                    torch.ones(1, device=f.device)).icdf(p0) 
                prob = prob*(1.0-base_distributions.Normal(torch.zeros(1, device=f.device),
                    torch.ones(1, device=f.device)).cdf(conf.mix_lr_ratio*f[...,i+self.num_copulas]+f0))

            mix.append(prob)

        stack_thetas = torch.stack(thetas)
        stack_mix = torch.stack(mix)

        assert torch.all(stack_thetas==stack_thetas)
        assert torch.all(stack_mix==stack_mix)
        return stack_thetas, stack_mix

    def fit(self, samples, f0 = None, n_epoch=200, lr=0.01):
        '''
        Using GPLink function as a parametrisation for copula parameters, 
        directly fit parameters of the corresponding copula to the data. 
        No GP involved. Use with caution (check plot_loss), 
        convergence rate differs from bvcopula/infer.py
        Parameters:
        ----------
        samples: Tensor
            The data
        f0: Tensor, optional
            The starting parameters in f-space (before GPLink)
        n_epoch: int
            Number of epochs
        lr: float
            Learning rate
        Returns:
        ----------
        best_copula: MixtureCopula
            A MixtureCopula model with the optimal parameters
        '''
        device = samples.device
        if f0 is None:
            f0 = torch.zeros((self.f_size,1),device=device)
        assert device == f0.device
        f = torch.autograd.Variable(f0, requires_grad = True) 
        optimizer = torch.optim.Adam([f], lr=lr)
        plot_loss = torch.zeros((n_epoch),device=device)
        for epoch in range(n_epoch):
            copula = self(f)
            loss = - copula.log_prob(samples).mean()
            if (loss<torch.min(plot_loss)) or (epoch==0):
                best_copula = self(f.detach())
            plot_loss[epoch] = loss.data
            loss.backward()
            grad = f.grad.data
            if torch.nonzero(grad!=grad).numel()!=0:
                print('NaN grad in f, fixing...')
                grad[grad!=grad] = 0
            optimizer.step()
        return best_copula#, plot_loss

    def forward(self, function_samples: Tensor, *params: Any) -> MixtureCopula:
        assert torch.all(function_samples==function_samples)
        thetas, mix = self.gplink_function(function_samples)
        return self.copula(thetas, 
                             mix, 
                             [lik.copula for lik in self.likelihoods], 
                             rotations=[lik.rotation for lik in self.likelihoods])
