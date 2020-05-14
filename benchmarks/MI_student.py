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
from benchmarks import train4entropy, integrate_student

NSamp=10000
device = torch.device('cuda:0')
x = torch.linspace(0.,1.,NSamp).numpy()
train_x = torch.tensor(x).float().to(device=device)

likelihoodC =  [bvcopula.GaussianCopula_Likelihood(),
                # bvcopula.GumbelCopula_Likelihood(rotation='0°'),
                bvcopula.GumbelCopula_Likelihood(rotation='180°')] 
likelihoodU =  [bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
				bvcopula.GumbelCopula_Likelihood(rotation='0°'),
				bvcopula.GaussianCopula_Likelihood()]

filename = "StudentH_8up.pkl"
mc_size = 2000
sem_tol=0.015
Rps = 5

#define functions
Frhos = lambda NN: np.ones(NN)*0.7
Fdfs = lambda NN: np.exp(5*np.linspace(0,1,NN))+1

rhos = Frhos(NSamp)
dfs = Fdfs(NSamp)

for Nvar in [8,9,10]:
	HRgS = utils.student_H(rhos,dfs,Nvar)/np.log(2)

	if Nvar<=5:
		v = False
	else:
		v = True

	print('Integrating true MI:')
	t1 = time.time()
	trueMI = integrate_student(Nvar,Frhos,Fdfs,sem_tol=sem_tol,verbose=v)
	t2 = time.time()
	print(f'Took {(t2-t1)//60} min {(t2-t1)%60:.0f} sec')

	for repetition in range(Rps):

		res = {}
		res['Nvar'] = Nvar
		res['NSamp'] = NSamp
		res['true_HRgS'] = HRgS
		res['true_integral'] = trueMI

		print(f"Nvar={Nvar}, {repetition+1}/{Rps}")

		t1 = time.time()

		y = utils.student_rvs(Nvar,rhos,dfs,1).squeeze()
		y0 = np.zeros_like(y)
		for i in range(y.shape[0]):
		    y0[i] = t.cdf(y[i],df=dfs[i])
		res['y'] = y
		res['y0'] = y0

		# run classic estimators
		res['BI-KSG_N'], res['BI-KSG_N_H'] = MI.BI_KSG(x.reshape((*x.shape,1)),y,)
		res['KSG_N'], res['KSG_N_H'] = MI.Mixed_KSG(x,y)
		res['BI-KSG'], res['BI-KSG_H'] = MI.BI_KSG(x.reshape((*x.shape,1)),y0,)
		res['KSG'], res['KSG_H'] = MI.Mixed_KSG(x,y0)

		# run neural network estimators
		for H in [100,200,500]: #H=1000 100% overfits
			mi = np.nan
			while mi!=mi:
				mi = MI.train_MINE(y0,H=H,lr=0.01,device=device).item()/np.log(2)
			res[f'MINE{H}'] = mi

		#now estimate
		# train conditional & unconditional CopulaGP 
		vine, eC = train4entropy(train_x,y0,likelihoodC,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v)
		_, eU = train4entropy(train_x,y0,likelihoodU,
			mc_size=mc_size,device=device,sem_tol=sem_tol,v=v,shuffle=True)
		res['eC'] = eC
		res['eU'] = eU

		#estimate Hr - Hrs
		res['estimated'] = (eU-eC).mean().item()
		#integrate a conditional copula
		subvine = vine.create_subvine(torch.arange(0,NSamp,50))
		CopulaGP = subvine.stimMI(s_mc_size=50, r_mc_size=20, sem_tol=sem_tol, v=v)
		res['integrated'] = CopulaGP[0].item()

		t2 = time.time()
		print(f"Took: {(t2-t1)//60} min")

		results_file = f"{home}/benchmarks/{filename}"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append(res)

		with open(results_file,"wb") as f:
			pkl.dump(results,f)		