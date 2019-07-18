import torch
from torch.distributions.transforms import Transform
from torch.distributions import constraints, normal

class NormTransform(Transform):
    """
    Transform via the mapping :math:`y = Normal.cdf(x)`.
    """
    domain = constraints.real
    codomain = constraints.interval(-1,1)
    bijective = False
    
    def __init__(self, cache_size=0):
        super(NormTransform, self).__init__(cache_size=cache_size)
        self.log_2_over_sqrt_pi = 0.12078223763524522234551844578164721225185272790259946836

    def __eq__(self, other):
        return isinstance(other, NormTransform)

    def _call(self, x):
        return normal.Normal(0,1).cdf(x) #torch.sigmoid(x)

    def _inverse(self, y):
        return normal.Normal(0,1).icdf(y) #self._logit(y)

    def log_abs_det_jacobian(self, x, y):
        # jacobian = 2/sqrt(pi) exp(-x**2)
        return -torch.pow(x,2) + self.log_2_over_sqrt_pi