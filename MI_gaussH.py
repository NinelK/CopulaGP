import pickle as pkl
import numpy as np

import os
import sys
home = '/home/nina/CopulaGP/'
sys.path.insert(0, home)

import torch
import utils
import marginal as mg
import time
import MI
from scipy.stats import t
from train import conf
import bvcopula#select_copula
from vine import CVine
from scipy.stats import sem as SEM

NSamp=10000
device = torch.device('cuda:1')
x = torch.linspace(0.,1.,NSamp).numpy()
train_x = torch.tensor(x).float().to(device=device)

lin_gauss = bvcopula.MixtureCopula(torch.linspace(-0.1,.9,NSamp,device=device).unsqueeze(0),
                    torch.ones(1,NSamp,device=device),
                    [bvcopula.GaussianCopula_Likelihood().copula])

def train4entropy(x,y,likelihood,shuffle=False):
    Nvar = y.shape[-1]
    data_layers = [torch.tensor(y).clamp(0.001,0.999).float().to(device)]
    copula_layers = []
    for m in range(0,Nvar-1):
        copulas, layer = [], []
        for n in range(1,Nvar-m):
            if shuffle:
                x = x[torch.randperm(NSamp)]
            samples = data_layers[-1][...,[n,0]]
            waic, model = bvcopula.infer(likelihood,x,samples,device=device) 
            print(f'{m},{n+m} WAIC: {waic:.3f}')
            if shuffle:
                x = x[torch.randperm(NSamp)]
            with torch.no_grad():
                f = model(x).mean
                copula = model.likelihood.get_copula(f)
                copulas.append(copula)
                layer.append(copula.ccdf(samples))
        data_layers.append(torch.stack(layer,dim=-1))
        copula_layers.append(copulas)
    print('Trained')
    vine_trained = CVine(copula_layers,x,device=device)
    entropies = vine_trained.entropy(sem_tol=0.01, mc_size=2500, v=v)
    return vine_trained, entropies

likelihoodTC = [bvcopula.GaussianCopula_Likelihood()] #True conditional (known)
likelihoodTU = [bvcopula.IndependenceCopula_Likelihood(),
				bvcopula.GaussianCopula_Likelihood(),
				bvcopula.GumbelCopula_Likelihood(rotation='0°'),
				bvcopula.GumbelCopula_Likelihood(rotation='180°')] # True unconditional
likelihoodC =  [bvcopula.GaussianCopula_Likelihood(),
				bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
				bvcopula.GumbelCopula_Likelihood(rotation='180°')]
likelihoodU =  [bvcopula.GaussianCopula_Likelihood(),
				bvcopula.GumbelCopula_Likelihood(rotation='0°'),
				bvcopula.GumbelCopula_Likelihood(rotation='180°')]

sem_tol = 0.01
Rps = 3

for Nvar in range(5,6):

	if Nvar<=5:
		v = False
	else:
		v = True

	for repetition in range(Rps):
		print(f"Nvar={Nvar}, {repetition+1}/{Rps}")

		t0 = time.time()

		#try to brute force the true value, if dimensionality still permits
		true_integral = 0
		copula_layers = [[lin_gauss for j in range(Nvar-1-i)] for i in range(Nvar-1)]
		vine = CVine(copula_layers,train_x,device=device)
		if Nvar<=5:
			subvine = vine.create_subvine(torch.arange(0,NSamp,10))
			a = True
			while a:
				CopulaGP = subvine.stimMI(s_mc_size=1000, r_mc_size=10, sem_tol=sem_tol, v=v)
				a = CopulaGP[1].item()!=CopulaGP[1].item()
			true_integral = CopulaGP[0].item()

		t01 = time.time()

		#sample
		y=vine.sample().cpu().numpy()
		#transform samples
		new_y = y.copy()
		new_y += np.repeat(y.prod(axis=-1).reshape(NSamp,1),Nvar,axis=-1)**(1/Nvar)
		transformed_y = (np.argsort(new_y.flatten()).argsort()/new_y.size).reshape(new_y.shape)

		#now estimate
		# train conditional & unconditional CopulaGP 
		_, eC = train4entropy(train_x,y,likelihoodTC)
		_, eU = train4entropy(train_x,y,likelihoodTU,shuffle=True)
		eT = vine.entropy(sem_tol=0.01, mc_size=2500, v=v)

		t1 = time.time()
		print(f"True value estimated in {(t1-t0)//60} min (int {(t01-t0)//60} min, est {(t1-t01)//60} min)")
		trueMI = [true_integral,(eU-eT).mean().item(),(eU-eC).mean().item()]

		# run classic estimators
		KSG = MI.BI_KSG(x.reshape((*x.shape,1)),new_y,)[0]
		Mixed_KSG = MI.Mixed_KSG(x,new_y)[0]
		KSG_T = MI.BI_KSG(x.reshape((*x.shape,1)),transformed_y,)[0]
		Mixed_KSG_T = MI.Mixed_KSG(x,transformed_y)[0]

		t2 = time.time()

		# run neural network estimators
		MINE = []
		for H in [50,100,200,500]: #H=1000 100% overfits
			mi = np.nan
			while mi!=mi:
				mi = MI.train_MINE(new_y,H=H,lr=0.01,device=device).item()/np.log(2)
			MINE.append(mi)
		t3 = time.time()

		# train conditional & unconditional CopulaGP 
		vineC, eC = train4entropy(train_x,transformed_y,likelihoodC)
		_, eU = train4entropy(train_x,transformed_y,likelihoodU,shuffle=True)

		#estimate Hr - Hrs
		estimated = (eU-eC).mean().item()
		#integrate a conditional copula
		integrated = 0
		if Nvar<=5:
			subvine = vineC.create_subvine(torch.arange(0,NSamp,10))
			CopulaGP = subvine.stimMI(s_mc_size=1000, r_mc_size=10, sem_tol=sem_tol, v=v)
			integrated = CopulaGP[0].item()

		t4 = time.time()
		print(f"Took: {(t4-t1)//60} min ({int(t2-t1)} sec, {(t3-t2)//60} min, {(t4-t3)//60} min)")

		res = [[Nvar,NSamp],trueMI,
			[[estimated, integrated], [KSG, KSG_T], [Mixed_KSG, Mixed_KSG_T], MINE],[y,transformed_y]]

		print(res[1])
		print(res[2])

		filename = "MI_sqGaussH_dump2.pkl"
		results_file = f"{home}/{filename}"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append(res)

		with open(results_file,"wb") as f:
			pkl.dump(results,f)	