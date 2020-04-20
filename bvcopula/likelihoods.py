import torch
from torch import Tensor
from typing import Any
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.distributions import MultivariateNormal, base_distributions
from gpytorch.utils.deprecation import _deprecate_kwarg_with_transform
from gpytorch.settings import num_gauss_hermite_locs
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

    @staticmethod
    def normalize(theta: Tensor) -> Tensor:
        return theta

class GaussianCopula_Likelihood(Copula_Likelihood_Base):
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = GaussianCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Gaussian'

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
    def __init__(self, **kwargs: Any):
        super(Copula_Likelihood_Base, self).__init__(**kwargs)
        self.copula = StudentTCopula
        self.rotation = None
        self.isrotatable = False
        self.name = 'Student T'

    @staticmethod
    def gplink_function(f: Tensor) -> Tensor:
        return torch.erf(f)

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
        if len(likelihoods)<=4:
            self.waic_samples = conf.waic_samples
        else:
            self.waic_samples = int(conf.waic_samples/2)
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
        Returns
            `MI` (Mutual information)
        '''

        with torch.no_grad():
            N = target[...,0].numel() #number of data points. The last dimension is (2,) here.
            samples_shape = torch.Size([self.waic_samples])
            f_samples = gp_distr.rsample(samples_shape) # [GP samples x GP variables x input shape]
            # from GP perspective it is [sample x batch x event] dims
            target = target.expand(samples_shape+target.shape)
            log_prob = self.get_copula(f_samples).log_prob(target).detach()
            pwaic = torch.var(log_prob,dim=0).sum()
            sum_prob = torch.exp(log_prob).sum(dim=0)
            S = torch.ones_like(pwaic)*self.waic_samples
            lpd=(sum_prob.log()-S.log()).sum() # sum_M log(1/N * sum^i_S p(y|theta_i)), where N is train_x.shape[0]

        if combine_terms:
            return -(lpd-pwaic)/N #=WAIC
        else:
            return lpd,pwaic

    def get_copula(self, f):
        '''
        Returns a copula given the GP sample
        '''
        thetas, mixes = self.gplink_function(f)
        copulas = [lik.copula for lik in self.likelihoods]
        rotations = [lik.rotation for lik in self.likelihoods]
        return self.copula(thetas,mixes,copulas,rotations=rotations)

    def stimMI(self, S, F, alpha=0.05, sem_tol=1e-3,
        s_mc_size=200, r_mc_size=20, sR_mc_size=5000):
        '''
        Estimates the mutual information between the stimulus 
        (conditioning variable) and the response (observable variables
        modelled with copula mixture) with the Robbins-Monro algorithm.
        Parameters
        ----------
        S: Tensor
            A representative set that defines the distribution of 
            the conditioning variable
        F: Tensor
            A batch of values from GP copula parameter priors,
            taken on a set S 
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.  (Default: 1e-3)
        s_mc_size : integer, optional
            Number of stimuli samples that are drawn from S in each iteration 
            of the MI estimation.  (Default: 200)
        r_mc_size : integer, optional
            Number of response samples that are drawn from a copula model for 
            a set of stimuli (size of s_mc_size) in each iteration of 
            the MI estimation.  (Default: 20)
        sR_mc_size : integer, optional            
            Number of stimuli samples that are drawn from S in each iteration 
            of the p(R) estimation. (Default: 5000)
        Returns
        -------
        ent : float
            Estimate of the MI in bits.
        sem : float
            Standard error of the MI estimate in bits.
        '''
        # figure out which device we are using
        assert F.dim() == S.dim()+2 # +f and copula dimensions
        assert F.shape[-1] == S.shape[0]
        f_mc_size = F.shape[0] # samples are already passed to this function
        if S.is_cuda:
            device = S.get_device()
            assert F.get_device()==device
        else:
            device = torch.device('cpu')
        # Gaussian confidence interval for sem_tol and level alpha
        conf = torch.erfinv(torch.tensor([1. - alpha])).to(device)
        sem = torch.ones(2,f_mc_size).to(device)*float('inf')
        Hrs = torch.zeros(f_mc_size).to(device) # sum of conditional entropies 
        Hr = torch.zeros(f_mc_size).to(device) # entropy of p(r)
        var_sum = torch.zeros(2,f_mc_size).to(device)
        log2 = torch.tensor([2.]).log().to(device)
        k = 0
        N = r_mc_size*s_mc_size
        with torch.no_grad():
            while torch.any(sem >= sem_tol):
                # Sample from p(s)
                subset = torch.randperm(torch.numel(S))[:s_mc_size]
                subset_S = S.view(-1)[subset]
                copula = self(F[...,subset]) #[f, stimuli(positions)]
                # Generate samples from p(r|s)*p(s)
                samples = copula.rsample(sample_shape = torch.Size([r_mc_size]))
                # these are samples for p(r|s) for each s
                # size [responses(samples), fs, stimuli(positions), 2] = [r,f,s,2]
                logpRgS = copula.log_prob(samples) / log2 # [r,f,s]
                assert torch.all(logpRgS==logpRgS)
                assert torch.all(logpRgS.abs()!=float("inf"))
                logpRgS = torch.einsum("ijk->ikj",logpRgS) # [r,s,f]
                logpRgS = logpRgS.reshape(-1,f_mc_size) # [N,f]

                # marginalise s (get p(r)) and reshape
                samples = torch.einsum("ijk...->ikj...",samples) # samples x Xs x fs x 2 = [r, s, f, 2]
                samples = samples.reshape(-1,*samples.shape[2:]) # (samples * Xs) x fs x 2 = [r*s, f, 2]
                samples = samples.unsqueeze(dim=-2) # (samples * Xs) x fs x 1 x 2 = [r*s, f, 1, 2]
                samples = samples.expand([N,f_mc_size,sR_mc_size,2]) # (samples * Xs) x fs x Xs x 2
                # now find E[p(r|s)] under p(s) with MC
                rR = torch.ones(N,f_mc_size).to(device)*float('inf')
                pR = torch.zeros(N,f_mc_size).to(device)
                var_sumR = torch.zeros(N,f_mc_size).to(device)
                kR = 0
                # print(f"Start calculating p(r) {k}")
                while torch.any(rR >= sem_tol): #relative error of p(r) = absolute error of log p(r)
                    new_subset = torch.randperm(torch.numel(S))[:sR_mc_size]
                    new_subset_S = S.view(-1)[new_subset]
                    new_copula = self(F[...,new_subset]) #[copulas, stimuli(positions)]
                    pRs = new_copula.log_prob(samples).exp() # [r from p(r),f,new_s] = [N,f,s]
                    kR += 1
                    # Monte-Carlo estimate of p(r)
                    pR += (pRs.mean(dim=-1) - pR) / kR # [N,f]
                    # Estimate standard error
                    var_sumR += ((pRs - pR.unsqueeze(-1)) ** 2).sum(dim=-1) # [N,f]
                    semR = conf * (var_sumR / (kR * sR_mc_size * (kR * sR_mc_size - 1))).pow(.5) 
                    rR = semR/pR #relative error
                # print(f"Finished in {kR} steps")

                logpR = pR.log() / log2 #[N,f]
                k += 1
                if k>100:
                 print('MC integral failed to converge')
                 break
                # Monte-Carlo estimate of MI
                #MI += (log2p.mean(dim=0) - MI) / k # mean over sample dimensions -> [f]
                Hrs += (logpRgS.mean(dim=0) - Hrs) / k # negative sum H(r|s) * p(s)
                Hr += (logpR.mean(dim=0) - Hr) / k # negative entropy H(r)
                # Estimate standard error
                var_sum[0] += ((logpRgS - Hrs) ** 2).sum(dim=0)
                var_sum[1] += ((logpR - Hr) ** 2).sum(dim=0)
                sem = conf * (var_sum / (k * N * (k * N - 1))).pow(.5)
                # print(f"{Hrs.mean().item():.3},{Hr.mean().item():.3},{(Hrs.mean()-Hr.mean()).item():.3},\
                #     {sem[0].max().item()/sem_tol:.3},{sem[1].max().item()/sem_tol:.3}") #balance convergence rates
        return (Hrs-Hr), (sem[0]**2+sem[1]**2).pow(.5), Hr, sem[1] #2nd arg is an error of sum

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, 
        weights=None, particles=torch.Size([0]),
        gh_locs=20, *params: Any, **kwargs: Any) -> Tensor:
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
            target = target.expand(torch.Size([gh_locs]) + target.shape)
            with num_gauss_hermite_locs(gh_locs):
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
        # f dimensions are [f_samples dim x GP variables dim x theta dim]
        # theta dimensions will be [copulas dim x f samples dim x theta dim], where
        # f samples dim is essentially an extra batch dimension
        # note that f samples dim may be empty
        assert num_copulas + num_indep_thetas - 1==f.shape[-2] #independent thetas + mixing concentrations - 1 (dependent)
        # WARNING: we assume here that there is 1 theta dim. Need to rethink it if there will be multiparametric copulas. 

        lr_ratio = .5 # lr_mix / lr_thetas

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
