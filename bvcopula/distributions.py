import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints, normal
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
    support = constraints.real
    
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
        log_prob = torch.zeros(self.theta.shape) # by default
        
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            log_prob = log_prob.cuda(device=get_cuda_device) #TODO try removing this, do not want unnecessary copying
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
    support = constraints.real
    
    def ppcf(self, samples):
        vals = samples[..., 0] #will stay this for self.theta == 0
        vals[..., self.theta != 0] = (-torch.log1p(samples[..., 0] * torch.expm1(-self.theta) \
                / (torch.exp(-self.theta * samples[..., 1]) \
                - samples[..., 0] * torch.expm1(-self.theta * samples[..., 1]))) \
                / self.theta)[..., self.theta != 0]
        return vals

    def log_prob(self, value):

        self.theta_thr = 17. #gplink ensures that it never exceeds this value

        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros(self.theta.shape) # by default
        
        mask = (self.theta != 0) & (self.theta.abs() < self.theta_thr)
        log_prob[..., mask] = torch.log(-self.theta * torch.expm1(-self.theta)
                          * torch.exp(-self.theta
                                   * (value[..., 0] + value[..., 1]))
                          / (torch.expm1(-self.theta)
                             + torch.expm1(-self.theta * value[..., 0])
                             * torch.expm1(-self.theta * value[..., 1])) ** 2)[..., mask]

        # now put everything out of range to -inf (which was most likely Nan otherwise)
        log_prob[mask & ((value[..., 0] <= 0) | (value[..., 1] <= 0) |
                (value[..., 0] >= 1) | (value[..., 1] >= 1))] = -float("Inf") 
        
        return log_prob

class ClaytonCopula(SingleParamCopulaBase):
    
    arg_constraints = {"theta": constraints.positive}
    support = constraints.real
    
    def ppcf(self, samples):
        vals = torch.zeros(samples.shape[:-1])
        thetas_ = self.theta.expand_as(vals)
        vals[thetas_==0] = samples[thetas_==0][..., 0] #for self.theta == 0
        gtz = torch.all(samples > 0.0, dim=-1) & (self.theta != 0)
        nonzero_theta = self.theta[self.theta != 0]
        vals[gtz] = (1 - samples[gtz][..., 1]**(-nonzero_theta) \
                + (samples[gtz][..., 0] * (samples[gtz][..., 1]**(1 + nonzero_theta))) \
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
