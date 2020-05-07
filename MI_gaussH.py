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
device = torch.device('cuda:0')
x = torch.linspace(0.,1.,NSamp).numpy()
train_x = torch.tensor(x).float().to(device=device)

lin_gauss = bvcopula.MixtureCopula(torch.linspace(-0.1,1.0,NSamp,device=device).unsqueeze(0),
                    torch.ones(1,NSamp,device=device),
                    [bvcopula.GaussianCopula_Likelihood().copula])
sem_tol = 0.01

for Nvar in range(8,9):

	for repetition in range(1):
		print(f"Nvar={Nvar}, {repetition+1}/3")

		t0 = time.time()

		copula_layers = [[lin_gauss for j in range(Nvar-1-i)] for i in range(Nvar-1)]
		vine = CVine(copula_layers,train_x,device=device)
		subvine = vine.create_subvine(torch.arange(0,NSamp,20))
		a = True
		while a:
			CopulaGP = subvine.stimMI(s_mc_size=200, r_mc_size=50, sR_mc_size=3000, sem_tol=sem_tol, v=True)
			a = CopulaGP[1].item()!=CopulaGP[1].item()
	
		y=vine.sample().cpu().numpy()
		del(vine,subvine)
		new_y = y.copy()
		new_y += np.repeat(y.prod(axis=-1).reshape(NSamp,1),Nvar,axis=-1)**(1/Nvar)
		transformed_y = (np.argsort(new_y.flatten()).argsort()/new_y.size).reshape(new_y.shape)

		t1 = time.time()
		print(f"True value estimated in {(t1-t0)//60} min")
		trueMI = [CopulaGP[0].item(),CopulaGP[1].item()]

		KSG = mg.revised_mi(x.reshape((*x.shape,1)),new_y,)[0]
		Mixed_KSG = MI.Mixed_KSG(x,new_y)/np.log(2)

		t2 = time.time()

		MINE = []
		for H in [50,100,200,500]:
			mi = np.nan
			while mi!=mi:
				mi = MI.train_MINE(new_y,H=H,lr=0.01,device=device).item()/np.log(2)
			MINE.append(mi)
		t3 = time.time()

		N = Nvar
		#redirect logging here
		exp_pref='benchmark'
		data_layers = [torch.tensor(transformed_y).clamp(0.001,0.999).float().to(device)]
		copula_layers = []
		for m in range(0,N-1):
		    out_dir = f'{conf.path2outputs}/{exp_pref}/layer{m}'
		    copulas, layer = [], []
		    for n in range(1,N-m):
		        print(m,n+m)
		        samples = data_layers[-1][...,[n,0]]
		        waic, model = bvcopula.infer([bvcopula.GaussianCopula_Likelihood(),
		                                        bvcopula.ClaytonCopula_Likelihood(rotation='180°')],
		                                     train_x,samples,device=device)
		#         (likelihoods, waic) = select_copula.select_with_heuristics(x,samples.cpu().numpy(),device,exp_pref,out_dir,m,n+m,
		#                                                                   train_x=train_x, train_y=samples)
		#         weights_file = f"{out_dir}/model_{exp_pref}_{m}-{m+n}.pth"
		#         model = utils.get_model(weights_file, likelihoods, device) 
		        with torch.no_grad():
		            f = model(train_x).mean
		            copula = model.likelihood.get_copula(f)
		            copulas.append(copula)
		            layer.append(copula.ccdf(samples))
		    data_layers.append(torch.stack(layer,dim=-1))
		    copula_layers.append(copulas)

		vine_trained = CVine(copula_layers,train_x,device=device)

		subvine = vine_trained.create_subvine(torch.arange(0,NSamp,20))
		a = True
		while a:
			CopulaGP = subvine.stimMI(s_mc_size=200, r_mc_size=50, sR_mc_size=3000, sem_tol=sem_tol, v=True)
			a = CopulaGP[1].item()!=CopulaGP[1].item()
		# CopulaGP = subvine.stimMI(s_mc_size=50, r_mc_size=20, sem_tol=0.01, v=True)

		t4 = time.time()
		print(f"Took: {(t4-t1)//60} min ({int(t2-t1)} sec, {(t3-t2)//60} min, {(t4-t3)//60} min)")

		res = [[Nvar,NSamp],trueMI,
			[CopulaGP[0].item(),KSG,Mixed_KSG,MINE],[y,transformed_y]]

		print(res[1])
		print(res[2])

		del(vine_trained,subvine)

		results_file = f"{home}/MI_gaussH_dump.pkl"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append(res)

		with open(f"{home}/MI_gaussH_dump.pkl","wb") as f:
			pkl.dump(results,f)	