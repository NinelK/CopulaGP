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
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) #variance does not matter much
        
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
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) #variance does not matter much
        
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
    def __init__(self, likelihood, num_tasks, prior_rbf_length=0.1, 
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
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) #variance does not matter much
        
        self.mean_module = gpytorch.means.ConstantMean(batch_size=num_tasks)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior, batch_size=num_tasks, ard_num_dims=1),
            batch_size=num_tasks, ard_num_dims=None
        )
        
        #we specify prior here
        lengthscale_prior = gpytorch.priors.NormalPrior(prior_rbf_length, 1.0) #variance does not matter much
        
        # Initialize lengthscale and outputscale to mean of priors
        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean

    def forward(self, x):
        mean = self.mean_module(x)  # Returns an n_data vec
        covar = self.covar_module(x)
        assert all(mean==mean)
        mmn = gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)
        return mmn