import torch
import bvcopula
from math import sqrt

class VineGP():
    '''
    This is a data class for regular vines
    '''
    def __init__(self, models):
        '''
        Parameters
        ----------
        models: list
            list of Pair_CopulaGP_data objects
            containing pair copula data
        '''
        self._validate(models)
        self.trees = models

    @staticmethod
    def _validate(models):
        # depth = len(models)
        N = len(models[0])
        for tree in models:
            assert N == len(tree)
            for bvcopula in tree:
                assert type(bvcopula) == bvcopula.Pair_CopulaGP_data
            N -= 1

    def serialize(self):
        # return data
        pass

    def deserialize(self, data):
        # self.trees = ...(data)
        pass

    def sample(self):
        # go through the trees and do smth
        pass

    def create_cvine(self, inputs, device=torch.device('cpu')):
        # static_trees = self.sample() # get copula models with no uncertainty in parameters
        # return CVine(static_trees,inputs,device=device)
        pass

class CVine():
    '''
    This class represents copula C-Vine
    '''
    def __init__(self, layers, inputs, device=torch.device('cpu')):
        '''
        Attributes
        ----------
        N: int
            Number of variables
        inputs: int
            Input points
        layers:
            A list of layers. Each layer should contain N-1-i
            MixtureCopula models, where i is the layer's number.
        '''
        super(CVine, self).__init__()
        # for N variables there must be N-1 layers
        self.N = len(layers[0])+1
        # get the inputs
        self.inputs = inputs
        for i,layer in enumerate(layers):
            assert len(layer) == self.N-1-i # check layer size
            for model in layer:
                assert (self.inputs.shape[0]== model.mix.shape[-1])
                assert (model.__class__ == bvcopula.distributions.MixtureCopula)
        self.layers = layers
        # ADD CHECK ON WHICH DEVICE EACH MODEL IS?
        self.device = device

    @classmethod
    def marginalize(cls,models_list,X,just_layers=False):
        '''
        This method takes a list of models (serialized),
        sequentially initialises Pair Copula-GP
        and marginalises the GP out
        '''
        # vine-type-indep
        copula_layers = []
        for layer in models_list:
            copula_layer = []
            for copula_mix in layer:
                copulaGP = copula_mix.model_init(X.device)
                copula = copulaGP.marginalize(X)
                copula_layer.append(copula)
            copula_layers.append(copula_layer)
        if just_layers:
            return copula_layers
        else:
            return cls(copula_layers,X,X.device) 
            
    @classmethod
    def mean(cls,models_list,X,device=torch.device("cpu"),just_layers=False):
        '''
        This method takes a list of models (serialized),
        sequentially initialises Pair Copula-GP
        and takes the mean GP parameters
        '''
        # vine-type-indep
        copula_layers = []
        for layer in models_list:
            copula_layer = []
            for copula_mix in layer:
                copulaGP = copula_mix.model_init(device)
                with torch.no_grad():
                    f = copulaGP.gp_model(X).mean
                copula = copulaGP.likelihood.get_copula(f) 
                copula_layer.append(copula)
            copula_layers.append(copula_layer)
        if just_layers:
            return copula_layers
        else:
            return cls(copula_layers,X,device) 

    @property
    def effective_dims(self):
        '''
        Calculates the effective dimensionality of the data.
        This method takes all non-independence elements in the vine,
        and uses this number `m` to estimate `dim = sqrt(2*m)`.
        '''
        all_c,ind_c = 0,0
        for tree in self.layers:
            for c in tree:
                if len(c.copulas) & (c.copulas[0].__name__=='IndependenceCopula'):
                    ind_c += 1
                all_c +=1
        return sqrt(2*(all_c-ind_c))

    def create_subvine(self, input_idxs: torch.Tensor):
        '''
        Creates a CVine object, defined on the subset of inputs
        input_idxs: torch.Tensor
            indexes of the input elements to keep
        '''
        assert input_idxs.max() < self.inputs.numel()
        new_layers = []
        for layer in self.layers:
            models = []
            for model in layer:
                if model.theta.numel()>0:
                    copula_model = bvcopula.MixtureCopula(model.theta[...,input_idxs],
                        model.mix[...,input_idxs],
                        model.copulas,
                        rotations=model.rotations)
                else:
                    #it is Independence
                    copula_model = bvcopula.MixtureCopula(model.theta,
                        model.mix[...,input_idxs],
                        model.copulas)
                models.append(copula_model)
            new_layers.append(models)
        return CVine(new_layers,self.inputs[input_idxs],device=self.device)

    def truncate(self, Ncut: int): 
        '''
        Creates a truncated vine, with the models in 
        the last Ncut layers replaced with Independence.
        Parameters
        ----------
        Ncut: int
            Number of layers to truncate
        Returns
        ----------
        truncated: CVine
            Truncated vine
        '''
        L = len(self.layers)
        assert (Ncut<=L) & (Ncut>0)
        truncated_layers = [[model for model in layer] for layer in self.layers[:Ncut]]
        return CVine(truncated_layers,self.inputs,device=self.device)
        
    @staticmethod
    def _layer_transform(upper,new,copulas):
        '''
        Parameters
        ----------
        upper: list
            Higher layer, which is already transformed
        new: torch.Tensor ?
            New variable to be added on this layer
        copulas: list
            List of MixtureCopula models for this layer
        '''
        assert upper.shape[-1] == len(copulas)
        lower_layer = [new]
        for n, copula in enumerate(copulas):
            stack = torch.stack([upper[...,n],new],dim=-1)
            lower_layer.append(copula.make_dependent(stack))
        return torch.einsum('i...->...i',torch.stack(lower_layer))

    def sample(self, sample_size = torch.Size([])):
        # create uniform samples
        samples_shape = self.inputs.shape + sample_size + torch.Size([self.N])
        samples = torch.empty(size=samples_shape, device=self.device).uniform_(1e-4, 1. - 1e-4) #torch.rand(shape) torch.rand in (0,1]
        
        missing_layers = self.N - 1 - len(self.layers)
        transformed_samples = [samples[...,-1-missing_layers:]]
        for copulas in self.layers[::-1]:
            upper = transformed_samples[-1]
            new_layer = self._layer_transform(upper,samples[...,self.N-upper.shape[-1]-1],copulas)
            transformed_samples.append(new_layer)
        
        return transformed_samples[-1]

    def log_prob(self, Y: torch.Tensor) -> torch.Tensor:
        
        assert Y.shape[-2] == self.inputs.shape[0]
        assert Y.shape[-1] == self.N
        layers = [Y] # zero layer
        log_prob = torch.zeros_like(Y[...,0])
        for layer, copulas in enumerate(self.layers):
            next_layer = []
            for n, copula in enumerate(copulas):
                #print(layer,layer+n+1, copula.copulas)
                log_prob += copula.log_prob(layers[-1][...,[n+1,0]])
                next_layer.append(copula.ccdf(layers[-1][...,[n+1,0]]))
            layers.append(torch.stack(next_layer,dim=-1))

        return log_prob

    def entropy(self, alpha=0.05, sem_tol=1e-3, mc_size=10000, v=False):
        '''
        Estimates the entropy of the mixture of copulas 
        with the Robbins-Monro algorithm.
        Parameters
        ----------
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.  (Default: 1e-3)
        mc_size : integer, optional
            Number of samples that are drawn in each iteration of the Monte
            Carlo estimation.  (Default: 10000)
        v : bool, default = False
            Verbose mode
        Returns
        -------
        ent : float
            Estimate of the entropy in bits.
        sem? : float
            Standard error of the entropy estimate in bits.
        '''

        # Gaussian confidence interval for sem_tol and level alpha
        conf = torch.erfinv(torch.tensor([1. - alpha],device=self.device))
        inputs = self.inputs.numel()
        sem = torch.ones(inputs,device=self.device)*float('inf')
        ent = torch.zeros(inputs,device=self.device) #theta here must have dims: copula x batch dims
        var_sum = torch.zeros(inputs,device=self.device)
        log2 = torch.tensor([2.],device=self.device).log()
        k = 0
        with torch.no_grad():
            while torch.any(sem >= sem_tol):
                # Generate samples
                samples = self.sample(torch.Size([mc_size])) # inputs (MC) x samples (MC) x variables
                samples = torch.einsum("ij...->ji...",samples) # samples (MC) x inputs (MC) x variables
                logp = self.log_prob(samples) # [sample dim, batch dims]
                assert torch.all(logp==logp)
                assert torch.all(logp.abs()!=float("inf")) #otherwise make masked tensor below
                log2p = logp / log2 #maybe should check for inf 2 lines earlier
                k += 1
                # Monte-Carlo estimate of entropy
                ent += (-log2p.mean(dim=0) - ent) / k # mean over samples dimension
                # Estimate standard error
                var_sum += ((-log2p - ent) ** 2).sum(dim=0)
                sem = conf * (var_sum / (k * mc_size * (k * mc_size - 1))).pow(.5)
                if v & (k%10==0):
                    print (sem.max()/sem_tol)
                if k>1000:
                    print('Failed to converge')
                    return ent #0
        return ent#, sem

       
    def inputMI(self, alpha=0.05, sem_tol=1e-2, 
          s_mc_size=200, r_mc_size=50, sR_mc_size=5000, v=False):
        '''
        Estimates the mutual information between the GP inputs
        and the outputs (observable variables
        modelled with copula mixture) with the Robbins-Monro algorithm.
        Parameters
        ----------
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.  (Default: 1e-3)
        s_mc_size : integer, optional
            Number of input samples that are drawn from S in each iteration 
            of the MI estimation.  (Default: 200)
        r_mc_size : integer, optional
            Number of output samples that are drawn from a copula model for 
            a set of inputs (size of s_mc_size) in each iteration of 
            the MI estimation.  (Default: 20)
        sR_mc_size : integer, optional            
            Number of input samples that are drawn from S in each iteration 
            of the p(R) estimation. (Default: 5000)
        v : bool, default = False
            Verbose mode
        Returns
        -------
        ent : float
            Estimate of the MI in bits.
        sem : float
            Standard error of the MI estimate in bits.
        '''

        # Notation here: X = S (stimuli), Y = R (response) 
        
        # Gaussian confidence interval for sem_tol and level alpha
        conf = torch.erfinv(torch.tensor([1. - alpha])).to(self.device)
        sem = torch.ones(2).to(self.device)*float('inf')
        Hrs = torch.zeros(1).to(self.device) # sum of conditional entropies 
        Hr = torch.zeros(1).to(self.device) # entropy of p(r)
        var_sum = torch.zeros_like(sem)
        log2 = torch.tensor([2.]).log().to(self.device)
        k = 0
        inputs = self.inputs.numel()
        if inputs<sR_mc_size:
            r_mc_size *= int(sR_mc_size/inputs)
            sem_tol_pr = sem_tol*int(sR_mc_size/inputs)
            sR_mc_size = inputs
        else:
            sem_tol_pr = sem_tol
        N = r_mc_size*s_mc_size
        max_log = 50.
        #88./(torch.tensor([sR_mc_size]).float().log()).item()
        with torch.no_grad():
            while torch.any(sem >= sem_tol):
                # Sample from p(s) (we can't fit all samples S into memory for this method)
                subset = torch.randperm(inputs)[:s_mc_size]
                subvine = self.create_subvine(subset)
                # Generate samples from p(r|s)*p(s)
                samples = subvine.sample(torch.Size([r_mc_size])) # inputs (MC) x responses (MC) x variables
                samples = torch.einsum("ij...->ji...",samples) # responses (MC) x inputs (MC) x variables
                # these are samples for p(r|s) for each s
                # size [responses(samples), stimuli(inputs), variables] = [r,s,v]
                logpRgS = subvine.log_prob(samples) / log2 # dim=-2 aligns with the number of inputs in subvine
                assert torch.all(logpRgS==logpRgS)
                assert torch.all(logpRgS.abs()!=float("inf"))
                logpRgS = logpRgS.reshape(-1) # [N]

                # marginalise s (get p(r)) and reshape
                samples = samples.reshape(-1,samples.shape[-1]) # (samples * Xs) x variables = [r*s, v]
                samples = samples.unsqueeze(dim=-2) # (samples * Xs) x 1 x 2 = [r*s, 1, v]
                samples = samples.expand([N,sR_mc_size,samples.shape[-1]]) # (samples * Xs) x Xs x v

                # now find E[p(r|s)] under p(s) with MC
                rR = torch.ones(N).to(self.device)*float('inf')
                pR = torch.zeros(N).to(self.device)
                var_sumR = torch.zeros(N).to(self.device)
                kR = 0
                if v:
                    print(f"Start calculating p(r) {k}")
                while torch.any(rR >= sem_tol_pr): #relative error of p(r) = absolute error of log p(r)
                    new_subset = torch.randperm(inputs)[:sR_mc_size] # permute samples & inputs
                    new_subvine = self.create_subvine(new_subset)
                    pRs = new_subvine.log_prob(samples).clamp(-float("inf"),max_log).exp()
                    assert torch.all(pRs==pRs)
                    kR += 1
                    # Monte-Carlo estimate of p(r)
                    pR += (pRs.mean(dim=-1) - pR) / kR
                    assert torch.all(pR==pR)
                    # Estimate standard error
                    var_sumR += ((pRs - pR.unsqueeze(-1)) ** 2).clamp(0,1e2).sum(dim=-1)
                    semR = conf * (var_sumR / (kR * sR_mc_size * (kR * sR_mc_size - 1))).pow(.5) 
                    rR = semR/pR #relative error
                    if v & (kR%100 == 0):
                        print(rR.max()/sem_tol_pr)
                    if kR>1000:
                        print('p(r) failed to converge')
                        break
                if v:
                    print(f"Finished in {kR} steps")

                assert torch.all(pR==pR)
                logpR = pR.log() / log2 #[N,f]
                k += 1
                if k>50:
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
                if v:
                    print(f"{Hrs.mean().item():.3},{Hr.mean().item():.3},{(Hrs.mean()-Hr.mean()).item():.3},\
                        {sem[0].max().item()/sem_tol:.3},{sem[1].max().item()/sem_tol:.3}") #balance convergence rates
        assert torch.all(Hr==Hr)
        return (Hrs-Hr), (sem[0]**2+sem[1]**2).pow(.5), Hrs, sem[1] #2nd arg is an error of sum