import torch
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints, normal
from torch.distributions.utils import _standard_normal

class GaussianCopula(TorchDistribution):
    
    has_rsample = True
    arg_constraints = {"theta": constraints.interval(-1,1)}
    support = constraints.real
    
    def __init__(self, theta, validate_args=None):
        self.theta = theta
        batch_shape, event_shape = self.theta.shape, torch.Size([2])
        super(GaussianCopula, self).__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianCopula, _instance)
        batch_shape = torch.Size(batch_shape)
        if batch_shape == torch.Size([]):
            batch_shape = torch.Size([1])
        theta_shape = batch_shape + torch.Size(self.event_shape[:-1])
        new.theta = self.theta.expand(theta_shape) # not sure if expand only batch dimension
        super(GaussianCopula, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new
    
    def ppcf(self, samples, theta):
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            nrvs = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).icdf(samples)
            vals = normal.Normal(torch.zeros(1).cuda(device=get_cuda_device),torch.ones(1).cuda(device=get_cuda_device)).cdf(nrvs[..., 0] * torch.sqrt(1 - self.theta**2) + 
                                 nrvs[..., 1] * self.theta)
        else:    
            nrvs = normal.Normal(0,1).icdf(samples)
            vals = normal.Normal(0,1).cdf(nrvs[..., 0] * torch.sqrt(1 - self.theta**2) + 
                                 nrvs[..., 1] * self.theta) # here in nrvs too close to 0 or 1, may go to -inf, resulting in more samples in the corner, then there should be
        return vals

    def rsample(self, sample_shape=torch.Size([])):
        shape = self._extended_shape(sample_shape) # now it is theta_size (batch) x sample_size x 2 (event)
        
        if sample_shape == torch.Size([]):   # not sure what to do with 1 sample
            shape = torch.Size([1]) + shape
            
        samples = torch.empty(size=shape).uniform_(1e-4, 1. - 1e-4) #torch.rand(shape) torch.rand in (0,1]
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            samples = samples.cuda(device=get_cuda_device)
        samples[...,0] = self.ppcf(samples, self.theta)
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        assert value.shape[-1] == 2 #check that the samples are pairs of variables
        log_prob = torch.zeros(self.theta.shape) # by default
        
        if self.theta.is_cuda:
            get_cuda_device = self.theta.get_device()
            log_prob = log_prob.cuda(device=get_cuda_device)
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


    def entropy(self):
        raise NotImplementedError