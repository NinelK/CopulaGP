import torch
import gpytorch
from torch import all, Size
from .variational_strategies import GridInterpolationVariationalStrategy

class GPInferenceModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, train_x, likelihood):
        # Define all the variational stuff
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=train_x.numel()
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, train_x, variational_distribution
        )
        
        # Standard initializtation
        super(GPInferenceModel, self).__init__(variational_strategy)
        self.likelihood = likelihood
        
        # Mean, covar
        self.mean_module = gpytorch.means.ConstantMean()
        
        #we specify prior here
        prior_rbf_length = 0.5 
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) 
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
        )
        
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        #self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        mean = self.mean_module(x)  # Returns an n_data vec
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class KISS_GPInferenceModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, likelihood, prior_rbf_length=0.1, grid_size=128, grid_bounds=[(0, 1)]):
        # Define all the variational stuff
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(grid_size)
        variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
            self, grid_size, grid_bounds, variational_distribution
        )
        
        # Standard initializtation
        super(KISS_GPInferenceModel, self).__init__(variational_strategy)
        self.likelihood = likelihood
        
        # Mean, covar
        self.mean_module = gpytorch.means.ConstantMean()
        
        #we specify prior here
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) 
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior),
        )
        
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        #self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        mean = self.mean_module(x)  # Returns an n_data vec
        covar = self.covar_module(x)
        assert all(mean==mean)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class Mixed_GPInferenceModel(gpytorch.models.AbstractVariationalGP):
    def __init__(self, likelihood, num_tasks, prior_rbf_length=0.5, 
                 grid_size = 128, grid_bonds = [(0, 1)]):
        # Define all the variational stuff
        variational_distribution = \
        gpytorch.variational.CholeskyVariationalDistribution(grid_size, 
                                                              batch_shape=Size([num_tasks]))
        
        variational_strategy = GridInterpolationVariationalStrategy(
            self,
            num_tasks,
            grid_size, 
            grid_bonds, 
            variational_distribution
        )
        
        # Standard initializtation
        super(Mixed_GPInferenceModel, self).__init__(variational_strategy) 
        self.likelihood = likelihood
        
        # Mean, covar
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, .2)
        mean_prior = None #gpytorch.priors.NormalPrior(0., .01)
        
        self.mean_module = gpytorch.means.ConstantMean(prior=mean_prior,batch_size=num_tasks)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior, batch_size=num_tasks, ard_num_dims=1),
            batch_size=num_tasks, ard_num_dims=None
        )
        
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean

    def entropy(self, alpha=0.05, sem_tol=1e-2, mc_size=10000, interv=torch.tensor([0.,1.])):
        '''
        Estimates the entropy of the mixed vine.
        Parameters
        ----------
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.  (Default: 1e-2)
        mc_size : integer, optional
            Number of samples that are drawn in each iteration of the Monte
            Carlo estimation.  (Default: 10000)
        Returns
        -------
        ent : float
            Estimate of the mixed vine entropy in bits.
        sem : float
            Standard error of the mixed vine entropy estimate in bits.
        '''
        # Gaussian confidence interval for sem_tol and level alpha
        conf = torch.distributions.normal.Normal(torch.zeros(1),torch.ones(1)).icdf(torch.tensor([1. - alpha]))
        sem = float('inf')
        ent = torch.zeros(1)
        var_sum = torch.zeros(1)
        log2 = torch.tensor([2.]).log()
        where_is_mean = self.mean_module(torch.zeros(1))
        if where_is_mean.is_cuda:
            device = where_is_mean.get_device()
            log2 = log2.cuda(device=device)
            ent = ent.cuda(device=device)
            points = torch.empty(size=torch.Size([mc_size])).cuda(device=device)
        k = 0
        with torch.no_grad():
            while sem >= sem_tol:
                # Generate samples
                points = points.uniform_(*interv)
                functions = self(points)
                with gpytorch.settings.num_likelihood_samples(1):
                    samples = self.likelihood(self(points)).rsample().squeeze()
                log_prob_lambda = lambda function_samples: self.likelihood.forward(function_samples).log_prob(samples) #TODO: decide where to clamp
                logp = self.likelihood.quadrature(log_prob_lambda, functions) 
                assert torch.all(logp==logp)
                log2p = logp[logp.abs()!=float("inf")] / log2
                k += 1
                # Monte-Carlo estimate of entropy
                ent += (-log2p.mean() - ent) / k
                # Estimate standard error
                var_sum += ((-log2p - ent) ** 2).sum()
                sem = conf * (var_sum / (k * mc_size * (k * mc_size - 1))).pow(.5)
        return ent, sem

    def forward(self, x):
        mean = self.mean_module(x)  # Returns an n_data vec
        covar = self.covar_module(x)
        assert all(mean==mean)
        mmn = gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)
        return mmn