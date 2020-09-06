import torch
import gpytorch
from torch import all, Size
from gpytorch.distributions import MultitaskMultivariateNormal
import math
from .likelihoods import MixtureCopula_Likelihood
from . import conf

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(0, 1), prior_rbf_length=0.5):

        def _grid_size(num_dim):
            if num_dim<4:
                grid_size = conf.grid_size
            else:
                grid_size = int(conf.grid_size/int(math.log(num_dim)/math.log(2)))
            return grid_size

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=_grid_size(num_dim), batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=_grid_size(num_dim), grid_bounds=[grid_bounds],
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

    def MI(self, points, alpha=0.05, sem_tol=1e-3, f_size=5, mc_size=10000):
        '''
        Measure mutual information between variables 
        (=negative conditioned copula entropy)
        Parameters
        ----------
        points: Tensor
            Input points where MI (-entropy) is estimated.
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.  (Default: 1e-3)
        mc_size : integer, optional
            Number of samples that are drawn in each iteration of the Monte
            Carlo estimation.  (Default: 10000)
        Returns
        -------
        MI_mean : float
            Estimate of the MI in bits for a copula, parameterised with a mean of GP.
        MIs_mean : float
            Estimate of the mean MI in bits for a copula, parameterised with a GP.
        MIs_std : float
            Estimate of the standard deviation of MI in bits for a copula, 
            parameterised with a GP.
        '''
        MIs = []
        with torch.no_grad():
            fs = self(points).rsample(torch.Size([f_size])) #[samples_f, copulas, positions]
        f_mean = self(points).mean.unsqueeze(0)
        # now add mean f to a set of f samples
        fs = torch.cat((fs,f_mean),0) #[samples_f + 1, copulas, positions]

        copula = self.likelihood(fs)
        MIs = copula.entropy()
        MI_mean = MIs[-1]
        MIs = MIs[:-1]

        return (MI_mean,MIs.mean(dim=0),MIs.std(dim=0)) 

class Pair_CopulaGP():
    def __init__(self, copulas: list, device='cpu'):

        self.__likelihood = MixtureCopula_Likelihood(copulas,
                particles=torch.Size([0])).to(device=device).float()

        self.__gp_model = MultitaskGPModel(self.__likelihood.f_size, 
            grid_bounds=(0, 1), prior_rbf_length=0.5).to(device=device).float()

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

# theta_sharing, num_fs = _get_theta_sharing(likelihoods, theta_sharing)
# def _get_theta_sharing(likelihoods, theta_sharing):
#     if theta_sharing is not None:
#         theta_sharing = theta_sharing
#         num_fs = len(likelihoods)+thetas_sharing.max().numpy() # indep_thetas + num_copulas - 1
#     else:
#         theta_sharing = torch.arange(0,len(likelihoods)).long()
#         num_fs = 2*len(likelihoods)-1
#     return theta_sharing, num_fs
