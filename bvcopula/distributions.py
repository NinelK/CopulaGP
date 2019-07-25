import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints, normal, studentT
from torch.distributions.utils import _standard_normal


class SingleParamCopulaBase(TorchDistribution):
    
    has_rsample = True
    
    def __init__(self, theta, validate_args=None):
        self.theta = theta
        batch_shape, event_shape = self.theta.shape, torch.Size([2])
        super(SingleParamCopulaBase, self).__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SingleParamCopulaBase, _instance)
        batch_shape = torch.Size(batch_shape)
        if batch_shape == torch.Size([]):
            batch_shape = torch.Size([1])
        theta_shape = batch_shape + torch.Size(self.event_shape[:-1])
        new.theta = self.theta.expand(theta_shape) 
        super(SingleParamCopulaBase, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    def ppcf(self, samples, theta):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size([])):
        shape = self._extended_shape(sample_shape) # now it is theta_size (batch) x sample_size x 2 (event)
        
        if sample_shape == torch.Size([]):   # not sure what to do with 1 sample
            shape = torch.Size([1]) + shape
            
        samples = torch.empty(size=shape).uniform_(1e-4, 1. - 1e-4) #torch.rand(shape) torch.rand in (0,1]
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            samples = samples.cuda(device=get_cuda_device)
        samples[...,0] = self.ppcf(samples)
        return samples

    def log_prob(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

class GaussianCopula(SingleParamCopulaBase):
    
    arg_constraints = {"theta": constraints.interval(-1,1)}
    support = constraints.interval(0,1)
    
    def ppcf(self, samples):
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            nrvs = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(samples)
            vals = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(nrvs[..., 0] * torch.sqrt(1 - self.theta**2) + 
                                 nrvs[..., 1] * self.theta)
        else:    
            nrvs = normal.Normal(0,1).icdf(samples)
            vals = normal.Normal(0,1).cdf(nrvs[..., 0] * torch.sqrt(1 - self.theta**2) + 
                                 nrvs[..., 1] * self.theta) 
        return vals

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros_like(self.theta) # by default 0 and already on a correct device

        # Check CUDA and make a Normal distribution
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            nrvs = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(value)
        else:
            nrvs = normal.Normal(0,1).icdf(value)
        
        thetas = self.theta
        
        log_prob[(thetas >= 1.0)  & ((value[..., 0] - value[..., 1]).abs() < 1e-4)]      = float("Inf") # u==v
        log_prob[(thetas <= -1.0) & ((value[..., 0] - 1 + value[..., 1]).abs() < 1e-4)]  = float("Inf") # u==1-v
        
        mask = (thetas < 1.0) & (thetas > -1.0)
        log_prob[..., mask] = (2 * thetas * nrvs[..., 0] * nrvs[..., 1] - thetas**2 \
            * (nrvs[..., 0]**2 + nrvs[..., 1]**2))[..., mask]
        log_prob[..., mask] /= 2 * (1 - thetas**2)[..., mask]
        log_prob[..., mask] -= torch.log(1 - thetas**2)[..., mask] / 2

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask & ((value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1))] = -float("Inf") 
        
        return log_prob

class FrankCopula(SingleParamCopulaBase):
    
    arg_constraints = {"theta": constraints.real}
    support = constraints.interval(0,1)
    
    def ppcf(self, samples):
        vals = samples[..., 0] #will stay this for self.theta == 0
        vals[..., self.theta != 0] = (-torch.log1p(samples[..., 0] * torch.expm1(-self.theta) \
                / (torch.exp(-self.theta * samples[..., 1]) \
                - samples[..., 0] * torch.expm1(-self.theta * samples[..., 1]))) \
                / self.theta)[..., self.theta != 0]
        return torch.clamp(vals,0.,1.) # could be slightly higher than 1 due to numerical errors

    def log_prob(self, value):

        self.theta_thr = 17.

        value[torch.isnan(value)] = 0 # log_prob = -inf

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros_like(self.theta) # by default 0 and already on a correct device
        
        mask = (self.theta.abs() > 1e-2) & (self.theta.abs() < self.theta_thr)
        theta_ = self.theta[mask]
        value_ = value.expand(self.theta.shape + torch.Size([2]))[mask]
        log_prob[..., mask] = torch.log(-theta_ * torch.expm1(-theta_)) \
                            - (theta_ * (value_[..., 0] + value_[..., 1])) \
                            - 2*torch.log(torch.abs(torch.expm1(-theta_)
                             + torch.expm1(-theta_ * value_[..., 0])
                             * torch.expm1(-theta_ * value_[..., 1])))

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask & ((value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1))] = -float("Inf") 

        assert torch.nonzero(torch.isnan(log_prob)).size(0) == 0 
        assert torch.nonzero(log_prob == +float("Inf")).size(0) == 0
        
        return log_prob

class ClaytonCopula(SingleParamCopulaBase):
    
    arg_constraints = {"theta": constraints.positive}
    support = constraints.interval(0,1)
    
    def ppcf(self, samples):
        vals = torch.zeros(samples.shape[:-1])
        thetas_ = self.theta.expand_as(vals)
        vals[thetas_==0] = samples[thetas_==0][..., 0] #for self.theta == 0
        nonzero_theta = thetas_[thetas_!=0]
        vals[thetas_!=0] = (1 - samples[thetas_!=0][..., 1]**(-nonzero_theta) \
                + (samples[thetas_!=0][..., 0] * (samples[thetas_!=0][..., 1]**(1 + nonzero_theta))) \
                ** (-nonzero_theta / (1 + nonzero_theta))) \
                ** (-1 / nonzero_theta)
        return vals

    def log_prob(self, value):

        self.exp_thr = torch.tensor(torch.finfo(torch.float32).max).log()

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros(self.theta.shape) # by default

        value_ = value.expand(self.theta.shape + torch.Size([2]))

        log_prob[value_[...,0] == value_[...,1]] = 1.

        log_prob[value_[...,0] == 1.] = (value_[...,1]**(1+self.theta))[value_[...,0] == 1.]
        log_prob[value_[...,1] == 1.] = (value_[...,0]**(1+self.theta))[value_[...,1] == 1.]
        
        log_base = -torch.min(value[...,0],value[...,1]).log() # max_theta depends on the coordinate of the value
        mask = (self.theta > 0) & (self.theta < self.exp_thr/log_base) \
                & (value[...,0] != value[...,1]) & torch.all(value < 1.0, dim=-1)
        log_prob[..., mask] = (torch.log(1 + self.theta) + (-1 - self.theta) \
                       * torch.log(value).sum(dim=-1) \
                       + (-1 / self.theta - 2) \
                       * torch.log(value_[...,0].pow(-self.theta) + value_[...,1].pow(-self.theta) - 1))[..., mask]

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask & ((value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] > 1) | (value[..., 1] > 1))] = -float("Inf") 
        
        return log_prob

class StudentTCopula(SingleParamCopulaBase):
    
    arg_constraints = {"theta": constraints.interval(-1,1)} 
    # here theta is actually Kendall's tau
    # it does not make sence to transform it through pi/2 * sin(tau) to get traditional theta
    support = constraints.interval(0,1)

    def __init__(self, theta, df = 2.0, validate_args=None):
        self.df = df #number of degrees of freedom (fixed here)
        assert(df == 2.0) #now only this one
        self.theta = theta
        batch_shape, event_shape = self.theta.shape, torch.Size([2])
        super(SingleParamCopulaBase, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @staticmethod
    def cdf_approx_2(x):
        return 0.5 + 0.5 * x / torch.sqrt(x**2 + 2.)

    @staticmethod
    def icdf_approx_2(y):
        return torch.sign(y-0.5) * torch.sqrt(2/(1/(2*y-1)**2 - 1))

    @staticmethod
    def icdf_approx_3(y):
        x = torch.sign(y-0.5) * torch.sqrt(2/(1/(2*y-1)**2 - 1)) #icdf_approx_2
        ar = torch.atan(x/torch.sqrt(torch.tensor([3.])))/2 # initial seed in form trigonometric
        PI=torch.acos(torch.Tensor([-1]))
        for _ in range(10):
            ar = PI * (y-0.5) - 0.5*torch.sin(2*ar)
        return torch.sqrt(torch.tensor([3.]))*torch.tan(ar)
    
    def ppcf(self, samples):
        '''
        all packages actually invert cdf, like R:
        https://github.com/SurajGupta/r-source/blob/master/src/nmath/qt.c
        which is not an option on GPU;
        There are many approximations for particular dfs, described in:
        William T. Shaw Sampling Student’s T distribution – use of the
        inverse cumulative distribution function
        He also describes a sampling method that oes not have to use icdf at all
        (Bailey’s method), which is also, unfortunatelly, iterative,
        meaning that it is not GPU-friendly.

        For now, we'll simply use an pproximation for df=2.0:
        cdf = 1/2 + x/2/sqrt(x**2+2)
        icdf = sign(x-0.5) * sqrt(2/(1/(2y-1)**2 - 1))
        '''
        kTM = torch.zeros_like(self.theta)
        kTM[self.theta<0] = 1
        h1 = 1.0 - torch.pow(self.theta, 2.0)
        nu1 = self.df + 1.0 # nu1 = theta[1] + 1.0
        dist1 = studentT.StudentT(df=self.df, scale=1.0, loc=0.0)
        #dist2 = studentT.StudentT(df=nu1, scale=1.0, loc=0.0)

        samples[...,0] = kTM + torch.sign(self.theta) * samples[...,0]  # TODO: check input bounds

        # inverse CDF yields quantiles
        x = self.icdf_approx_3(samples[...,0])   #dist2
        y = self.icdf_approx_2(samples[...,1])   #dist1

        # eval H function
        vals = self.cdf_approx_2(x * torch.sqrt((self.df + torch.pow(y, 2.0)) * h1 / nu1) + self.theta * y)
        return vals

    def log_prob(self, value):

        self.exp_thr = torch.tensor(torch.finfo(torch.float32).max).log()

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros(self.theta.shape) # by default

        value_ = value.expand(self.theta.shape + torch.Size([2]))

        log_prob[value_[...,0] == value_[...,1]] = 1.

        log_prob[value_[...,0] == 1.] = (value_[...,1]**(1+self.theta))[value_[...,0] == 1.]
        log_prob[value_[...,1] == 1.] = (value_[...,0]**(1+self.theta))[value_[...,1] == 1.]
        
        log_base = -torch.min(value[...,0],value[...,1]).log() # max_theta depends on the coordinate of the value
        mask = (self.theta > 0) & (self.theta < self.exp_thr/log_base) \
                & (value[...,0] != value[...,1]) & torch.all(value < 1.0, dim=-1)
        log_prob[..., mask] = (torch.log(1 + self.theta) + (-1 - self.theta) \
                       * torch.log(value).sum(dim=-1) \
                       + (-1 / self.theta - 2) \
                       * torch.log(value_[...,0].pow(-self.theta) + value_[...,1].pow(-self.theta) - 1))[..., mask]

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask & ((value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] > 1) | (value[..., 1] > 1))] = -float("Inf") 
        
        return log_prob
