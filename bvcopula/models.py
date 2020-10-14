import torch
import gpytorch
from torch import all, Size
from gpytorch.distributions import MultitaskMultivariateNormal
import math
from .likelihoods import MixtureCopula_Likelihood
from . import conf

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(0, 1), prior_rbf_length=0.5, grid_size=None):

        def _grid_size(num_dim):
            if num_dim<4:
                grid_size = conf.grid_size
            else:
                grid_size = int(conf.grid_size/int(math.sqrt(num_dim))) # ~constant memory usage for covar
            return grid_size

        if grid_size is None:
            self.grid_size = _grid_size(num_dim)
        else:
            assert type(grid_size) == int
            self.grid_size = grid_size

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=self.grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        #we specify prior here
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) 
        # gpytorch.priors.SmoothedBoxPrior(
        #             math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
        #         ),

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                batch_shape=torch.Size([num_dim])
            ),
            batch_shape=torch.Size([num_dim])
        )
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_dim]))
        self.grid_bounds = grid_bounds

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean = self.mean_module(x)  # Returns (num_indep_tasks=batch) x N matrix
        covar = self.covar_module(x) # batch x N x N
        assert all(mean==mean)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class Pair_CopulaGP():
    def __init__(self, copulas: list, device='cpu', grid_size=None):

        self.__likelihood = MixtureCopula_Likelihood(copulas).to(device=device).float()

        self.__gp_model = MultitaskGPModel(self.__likelihood.f_size, 
            grid_bounds=(0, 1), prior_rbf_length=0.5, grid_size=grid_size).to(device=device).float()

        self.__device = device

    @property
    def gp_model(self):
        return self.__gp_model

    @property
    def likelihood(self):
        return self.__likelihood

    @property
    def gplink(self):
        return self.__likelihood.gplink_function

    @property
    def device(self):
        return self.__device
    #TODO: mb add setter for device, that relocates all parts of the model?
