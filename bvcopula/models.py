import torch
import gpytorch
from torch import all, Size
from gpytorch.distributions import MultitaskMultivariateNormal
import math
from collections import OrderedDict
from .likelihoods import MixtureCopula_Likelihood
from . import conf
from .infer import infer

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
    def __init__(self, copulas: list, device='cpu', grid_size=None, prior_rbf_length=0.5):

        self.__likelihood = MixtureCopula_Likelihood(copulas).to(device=device).float()

        self.__gp_model = MultitaskGPModel(self.__likelihood.f_size, 
            grid_bounds=(0, 1), prior_rbf_length=prior_rbf_length, grid_size=grid_size).to(device=device).float()

        self.__device = device
        self.__particles = 50

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
    #TODO: particle number setter

    def cpu(self):
        return self.serialize().model_init(device=torch.device('cpu'))

    def marginalize(self,X: torch.Tensor):
        '''
        Marginalizes GP variable f out
        To do this, samples N samples from GP.
        For each input point, uses one of those N samples
        to parameterize a copula.
        Parameters
        ----------
        X: Tensor
            inputs
        Returns
        -------
        copula: MixtureCopula
            copula defined on X with f 'integrated' out
        '''

        with torch.no_grad():
            f = self.__gp_model(X).rsample(torch.Size([self.__particles])) #TODO particle num to conf
        f = torch.einsum('i...->...i', f)
        onehot = torch.rand(f.shape,device=self.__device).argsort(dim = -1) == 0
        f_samples = f[onehot].reshape(f.shape[:-1])

        return self.__likelihood.get_copula(f_samples) 

    def serialize(self):
        bvcopulas = self.__likelihood.serialize()    
        state_dict = self.__gp_model.state_dict()
        cpu_state_dict = OrderedDict({k: state_dict[k].cpu() for k in state_dict})
        return Pair_CopulaGP_data(bvcopulas, cpu_state_dict)

    def ablate(self,train_x,train_y):
        '''
        Ablates likelihood elements 1 by 1
        and infers the model parameters.
        Parameters
        ----------
        train_x: Tensor
        train_y: Tensor
        Returns
        -------
        waics:  list
            list of model waics
        models: list
            list of corresponding serialized models
        '''
        device = train_x.device
        assert train_y.device==device
        likelihoods = self.__likelihood.likelihoods
        N = len(likelihoods)
        waics, models = [], []
        for i in range(N):
            lls = [likelihoods[j] for j in torch.arange(N)[torch.arange(N)!=i]]
            w, m = infer(lls,train_x,train_y,device=device)
            waics.append(w)
            models.append(m.serialize())
        return (waics,models)

class Pair_CopulaGP_data():
    '''
    Data class for Pair Copula GP.
    Intended for structured storage of the models.
    '''
    def __init__(self, bvcopulas, weights):
        '''
        Init
        Parameters
        ----------
        bvcopulas: list
            a list of serialised bvcopulas
        weights: OrderedDict
            serialized weights of the model
        '''
        if weights is None:
            assert (len(bvcopulas) == 1) & (bvcopulas[0][0] == 'Independence')
        self.bvcopulas = bvcopulas
        self.weights = weights

    @property
    def name_string(self):
        '''
        Gets a string with the unique mixture name 
        (incl. rotations)
        '''
        strrot = lambda rotation: rotation if rotation is not None else ''
        copula_names=''
        for lik in self.bvcopulas:
            copula_names += lik[0]+strrot(lik[1])
        return copula_names

    def model_init(self, device):
        '''
        Spawns an instance of a Pair Copula GP model class,
        that can be used for compulations
        '''
        likelihoods = MixtureCopula_Likelihood.deserialize(self.bvcopulas,just_likelihoods=True)
        model = Pair_CopulaGP(likelihoods,device=device)
        if self.weights!=None:
            model.gp_model.load_state_dict(self.weights)
        return model