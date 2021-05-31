import pickle as pkl
import numpy as np

import os

import torch
import copulagp.utils as utils
import copulagp.marginal as mg
import time
import copulagp.MI as MI
from scipy.stats import t
import copulagp.bvcopula as bvcopula
from copulagp.vine import CVine
from scipy.stats import sem as SEM
from training import train4entropy


NSamp=10000
device = torch.device('cuda:1')
filename = "new_GaussH.pkl"
Nvars = [2,3,4,5,6,7,8,9,10]
mc_size = 1000

x = torch.linspace(0.,1.,NSamp).numpy()
train_x = torch.tensor(x).float().to(device=device)


def const_rho_layers(rho,Nvar):
    def lin_gauss(rhos):
        return bvcopula.MixtureCopula(rhos, 
                    torch.ones(1,NSamp,device=device), 
                    [bvcopula.GaussianCopula_Likelihood().copula])
    copula_layers = []
    rho_cond = rho.clone()
    for i in range(Nvar-1):
        copula_layers.append([lin_gauss(rho_cond) for j in range(Nvar-1-i)])
        rho_cond = (rho_cond - rho_cond**2) / (1 - rho_cond**2)
    return copula_layers

rho0 = torch.linspace(-0.1,.999,NSamp,device=device).unsqueeze(0)

# lin_clayton = bvcopula.MixtureCopula(torch.linspace(0.01,2.,NSamp,device=device).unsqueeze(0),
#                     torch.ones(1,NSamp,device=device),
#                     [bvcopula.ClaytonCopula_Likelihood().copula])

# likelihoodC = [bvcopula.ClaytonCopula_Likelihood()] #True conditional (known)
# likelihoodU = [bvcopula.FrankCopula_Likelihood(),
#                 bvcopula.ClaytonCopula_Likelihood(),] # True unconditional

likelihoodC = [bvcopula.GaussianCopula_Likelihood()] #True conditional (known)
likelihoodU = [bvcopula.IndependenceCopula_Likelihood(),
                bvcopula.GaussianCopula_Likelihood(),
                bvcopula.GumbelCopula_Likelihood(rotation='0°'),
                bvcopula.GumbelCopula_Likelihood(rotation='180°')] # True unconditional

sem_tol_base = 0.05
Rps = 3

for Nvar in Nvars:

	sem_tol = sem_tol_base #* Nvar

	if Nvar<=5:
		v = False
	else:
		v = True

	for repetition in range(Rps):

		res = {}
		res['Nvar'] = Nvar
		res['NSamp'] = NSamp

		print(f"Nvar={Nvar}, {repetition+1}/{Rps}")

		t0 = time.time()

		#try to brute force the true value, if dimensionality still permits
		copula_layers = const_rho_layers(rho0,Nvar)
		#[[lin_clayton for j in range(Nvar-1-i)] for i in range(Nvar-1)]
		vine = CVine(copula_layers,train_x,device=device)
		subvine = vine.create_subvine(torch.arange(0,NSamp,10))
		a = True
		while a:
			CopulaGP = subvine.inputMI(s_mc_size=50, r_mc_size=20, sem_tol=sem_tol, v=v)
			a = CopulaGP[1].item()!=CopulaGP[1].item()
		res['true_integral'] = CopulaGP[0].item()

		t01 = time.time()

		#sample
		y=vine.sample().cpu().numpy()
		res['y'] = y

		#now estimate
		# train conditional & unconditional CopulaGP 
		vineC, eC = train4entropy(train_x,y,likelihoodC,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v)
		_, eU = train4entropy(train_x,y,likelihoodU,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v,shuffle=True)
		eT = vine.entropy(sem_tol=sem_tol, mc_size=mc_size, v=v)
		res['eC'] = eC.cpu().numpy()
		res['eU'] = eU.cpu().numpy()
		res['eT'] = eT.cpu().numpy()

		t1 = time.time()
		print(f"True value estimated in {(t1-t0)//60} min (int {(t01-t0)//60} min, est {(t1-t01)//60} min)")

		# run classic estimators
		res['BI-KSG'], res['BI-KSG_H'] = MI.BI_KSG(x.reshape((*x.shape,1)),y,)
		res['KSG'], res['KSG_H'] = MI.Mixed_KSG(x,y)

		print(f"{res['true_integral']:.3f}, {(eU-eT).mean().item():.3f} ({eU.std().item():.3f}), \
{(eU-eC).mean().item():.3f} ({eU.std().item():.3f}), {res['BI-KSG']:.3f}, {res['KSG']:.3f}")

		t2 = time.time()

		#estimate Hr - Hrs
		estimated = (eU-eC).mean().item()
		res['estimated'] = estimated
		#integrate a conditional copula
		subvine = vineC.create_subvine(torch.arange(0,NSamp,10))
		CopulaGP = subvine.stimMI(s_mc_size=50, r_mc_size=20, sem_tol=sem_tol, v=v)
		res['integrated'] = CopulaGP[0].item()

		t3 = time.time()

		# run neural network estimators
		for H in [100,200,500]: #H=1000 100% overfits
			mi = np.nan
			while mi!=mi:
				mi = MI.train_MINE(y,H=H,lr=0.01,device=device).item()/np.log(2)
			res[f'MINE{H}'] = mi

		t4 = time.time()
		print(f"Took: {(t4-t0)//60} min")

		print(f"MI: {res['integrated']:.3f}, {res['estimated']:.3f} ({eU.std().item():.3f}), {res['BI-KSG']:.3f}, {res['KSG']:.3f}, {res['MINE100']:.3f}")
		print(f"H:, {-eT.mean().item():.3f}, {-eC.mean().item():.3f}, {-res['BI-KSG_H']:.3f}")

		results_file = f"{filename}"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append(res)

		with open(results_file,"wb") as f:
			pkl.dump(results,f)	
