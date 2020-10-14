import pickle as pkl
import numpy as np
from typing import Callable
import torch
from bvcopula import MixtureCopula, IndependenceCopula
from utils import get_model
from . import CVine

def load(models_lists: Callable[[], str], weight_files: Callable[[], str], 
              train_x: torch.tensor, gp_particles = torch.Size([])) -> CVine:
	'''
	Loads a vine model
	Parameters
	----------
	models_list: an str-valued function that returns paths
				to bivariate model lists
	weight_files: an str-valued function that returns paths
				to bivariate model weights
	train_x: (torch.Tensor)
			the domain X, on which the conditional vine
			p(Y|X) will be defined
	gp_particles: (torch.Size, default=torch.Size([]))
			the number of particles used for sampling from GPs.
			If empty: the mean of GP is taken
	Returns
	--------
	likelihoods: (list)
			A list of copula trees [0,1,2,3....], 
			each containing a list of bivariate copula 
			models of length (N,N-1,N-2,N-3,...),
			where bvcopula model is a mixture, 
			represented as a list of copula likelihoods.
			(so, 3-times nested list)
			If only a few 
	CVine: (bvcopula.CVine)
			A vine model
	'''
	N_points = train_x.numel()
	device = train_x.device

	with open(models_lists(0),"rb") as f:
	    results = pkl.load(f)
	NN = len(results)+1

	copula_layers, likelihoods, fs_layers = [], [], []
	for layer in range(0,NN-1):
	    copulas, fs = [], []
	    try:
	        with open(models_lists(layer),"rb") as f:
	            results = pkl.load(f)
	    except FileNotFoundError as er:
	#             print(f"Filling in the T{layer} with independence models")
	        for n in range(NN-1-layer):
	            copulas.append(MixtureCopula(torch.empty(1,0,device=device),
	                        torch.ones(1,N_points,device=device),
	                        [IndependenceCopula]))
	            fs.append(None)
	    else:
	        likelihoods.append([a[0] for a in results])
	        assert len(likelihoods[-1])==(NN-layer-1)
	        for n,res in enumerate(results):
	            if res[1]!='Independence':
	                model = get_model(weight_files(layer,n), likelihoods[layer][n], device)
	                with torch.no_grad():
	                    if gp_particles == torch.Size([]):
	                        f = model.gp_model(train_x).mean
	                    else:
	                        f0 = model.gp_model(train_x).rsample(gp_particles)
	                        f0 = torch.einsum('i...->...i', f0)
	                        onehot = torch.rand(f0.shape,device=f0.device).argsort(dim = -1) == 0
	                        f = f0[onehot].reshape(f0.shape[:-1])
	                    copula = model.likelihood.get_copula(f)
	                    copulas.append(copula)
	                    fs.append(f)
	            else:
	                copulas.append(MixtureCopula(torch.empty(1,0,device=device),
	                        torch.ones(1,N_points,device=device),
	                        [IndependenceCopula]))
	                fs.append(None)

	    copula_layers.append(copulas)
	    fs_layers.append(fs)
	return likelihoods, CVine(copula_layers,train_x,device=device)

def WAICs(models_lists: Callable[[], str]) -> np.ndarray:
	'''
	Returns bivariate likelihoods and WAICs of the corresponding
	models for all trees of the vine copulas.

	Parameters
	----------
	models_lists: (function) 
			an str-valued function that returns
		    a path to models list for each vine tree
	Returns
	----------
	WAICs: (np.ndarray)
			an NxN array, containing WAIC of the
			bivariate copula models in an upper triangle 		 
	'''
	assert type(models_lists(0)) == str
	with open(models_lists(0),"rb") as f:
	    results = pkl.load(f)
	NN = len(results)+1
	WAICs = np.zeros((NN,NN))
	n_field = 2 #the field that contains WAICs
	WAICs[0,1:] = [a[n_field] for a in results]
	for tree in range(1,len(results)):
	    try:
	        with open(models_lists(tree),"rb") as f:
	            res = pkl.load(f)
	    except FileNotFoundError as er:
	        print(er)
	        print('Loading stops and only the WAICs for lower trees are returned')
	        break
	    else:
	        WAICs[tree,(tree+1):] = [a[n_field] for a in res]
	return WAICs


####################################################
# Train static copula
#####################################################
# import time
# X,Y = utils.standard_loader(f"{conf.path2data}/{exp_pref}_standard.pkl")
# indep = bvcopula.MixtureCopula(torch.empty(1,0,device=device),
#                     torch.ones(1,1,device=device),
#                     [bvcopula.IndependenceCopula])
# N = Y.shape[-1]
# # device = torch.device('cpu')
# data_layers = [torch.tensor(Y).float().to(device)]
# copula_layers = []
# t0 = time.time()
# for m in range(0,N-1):
#     copulas, layer = [], []
#     for n in tqdm.tqdm(range(1,N-m)):
#         samples = data_layers[-1][...,[n,0]]
#         likelihood = likelihood_layers[m][n-1]
#         f0 = fs_layers[m][n-1]
#         if f0 is None:
#             assert likelihood[0].name=='Independence'
#             copulas.append(indep)
#             layer.append(samples[:,0])
#         else:
#             f0 = f0.mean(axis=-1).unsqueeze(-1)
# #             copula0 = likelihood(f0)
#             copula = likelihood.fit(samples,f0,n_epoch=500)
# #             print(f"{m},{n+m}: {(copula0.theta-copula.theta).mean().cpu()}")
#             copulas.append(copula)
#             layer.append(copula.ccdf(samples.unsqueeze(-2)).squeeze())
#     data_layers.append(torch.stack(layer,dim=-1))
#     copula_layers.append(copulas)
# t1= time.time()
# print(f"{(t1-t0)//60}")