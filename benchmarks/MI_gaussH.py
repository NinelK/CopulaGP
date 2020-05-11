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
import bvcopula#select_copula
from vine import CVine
from scipy.stats import sem as SEM
from benchmarks import train4entropy

NSamp=10000
device = torch.device('cuda:1')
x = torch.linspace(0.,1.,NSamp).numpy()
train_x = torch.tensor(x).float().to(device=device)

lin_gauss = bvcopula.MixtureCopula(torch.linspace(-0.1,.9,NSamp,device=device).unsqueeze(0),
                    torch.ones(1,NSamp,device=device),
                    [bvcopula.GaussianCopula_Likelihood().copula])

likelihoodTC = [bvcopula.GaussianCopula_Likelihood()] #True conditional (known)
likelihoodTU = [bvcopula.IndependenceCopula_Likelihood(),
                bvcopula.GaussianCopula_Likelihood(),
                bvcopula.GumbelCopula_Likelihood(rotation='0°'),
                bvcopula.GumbelCopula_Likelihood(rotation='180°')] # True unconditional
#rt
likelihoodC =  [bvcopula.GaussianCopula_Likelihood(),
                bvcopula.GumbelCopula_Likelihood(rotation='180°'),
                bvcopula.GumbelCopula_Likelihood(rotation='0°')]
likelihoodU =  [bvcopula.GaussianCopula_Likelihood(),
                bvcopula.GumbelCopula_Likelihood(rotation='180°'),
                bvcopula.GumbelCopula_Likelihood(rotation='0°')] 
# #full
# likelihoodC =  [bvcopula.GaussianCopula_Likelihood(),
# 				bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
#                 bvcopula.GumbelCopula_Likelihood(rotation='0°')]
# likelihoodU =  [bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
# 				bvcopula.GaussianCopula_Likelihood(),
#                 bvcopula.ClaytonCopula_Likelihood(rotation='180°')]

filename = "MI_rtGaussH_dump.pkl"
mc_size = 4000
sem_tol = 0.01
Rps = 1

for Nvar in [2,10]:

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
		_, eC = train4entropy(train_x,y,likelihoodTC,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v)
		_, eU = train4entropy(train_x,y,likelihoodTU,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v,shuffle=True)
		eT = vine.entropy(sem_tol=sem_tol, mc_size=mc_size, v=v)

		t1 = time.time()
		print(f"True value estimated in {(t1-t0)//60} min (int {(t01-t0)//60} min, est {(t1-t01)//60} min)")
		trueMI = [true_integral,(eU-eT).mean().item(),(eU-eC).mean().item()]
		print(trueMI)
		del(vine,subvine)

		# run classic estimators
		KSG = MI.BI_KSG(x.reshape((*x.shape,1)),new_y,)
		Mixed_KSG = MI.Mixed_KSG(x,new_y)
		KSG_T = MI.BI_KSG(x.reshape((*x.shape,1)),transformed_y,)
		Mixed_KSG_T = MI.Mixed_KSG(x,transformed_y)

		t2 = time.time()

		# run neural network estimators
		MINE = []
		for H in [50,100,200,500,1000]: #H=1000 100% overfits
			mi = np.nan
			while mi!=mi:
				mi = MI.train_MINE(new_y,H=H,lr=0.01,device=device).item()/np.log(2)
			MINE.append(mi)
		t3 = time.time()

		# train conditional & unconditional CopulaGP 
		vine, eC = train4entropy(train_x,transformed_y,likelihoodC,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v)
		_, eU = train4entropy(train_x,transformed_y,likelihoodU,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v,shuffle=True)

		#estimate Hr - Hrs
		estimated = (eU-eC).mean().item()
		#integrate a conditional copula
		integrated = 0
		if Nvar<=5:
			subvine = vine.create_subvine(torch.arange(0,NSamp,50))
			CopulaGP = subvine.stimMI(s_mc_size=50, r_mc_size=20, sem_tol=sem_tol, v=v)
			integrated = CopulaGP[0].item()

		del(vine,subvine)

		t4 = time.time()
		print(f"Took: {(t4-t1)//60} min ({int(t2-t1)} sec, {(t3-t2)//60} min, {(t4-t3)//60} min)")

		res = [[Nvar,NSamp],
				[trueMI,  KSG[0], Mixed_KSG[0]],
			[[integrated, estimated, KSG_T[0], Mixed_KSG_T[0]], MINE],
			[-eC.mean().item(),-KSG_T[1], -Mixed_KSG_T[1]],
			[y,transformed_y]]

		print(res[1])
		print(res[2])
		print(res[3])

		results_file = f"{home}/benchmarks/{filename}"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append(res)

		with open(results_file,"wb") as f:
			pkl.dump(results,f)	