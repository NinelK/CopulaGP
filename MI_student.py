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

rhos = np.ones(NSamp)*0.7
dfs = np.exp(5*np.linspace(0,1,NSamp))+1

for Nvar in [15,20]:
	HRgS = utils.student_H(rhos,dfs,Nvar)/np.log(2)

	NN=1000 #number of inputs
	N=10 #number of samples for each input
	M=1000 #number of permutation
	sem_tol=0.01

	rhos = np.ones(NN)*0.7
	dfs = np.exp(5*np.linspace(0,1,NN))+1

	sem = float("inf")
	HRS,HR, var_sum, k = 0,0,0,0
	while sem>sem_tol:
	    samples = utils.student_rvs(Nvar,rhos,dfs,N)
	    samples = np.einsum("ij...->ji...",samples)
	    log_probs = utils.student_logprob(samples,rhos,dfs)
	    Hrs = log_probs.mean(axis=-1)
	    Hr = np.zeros((N,NN))
	    v = np.zeros((N,NN))
	    tol = float("inf")
	    kR=0
	    t1 = time.time()
	    while tol>sem_tol:
	        new_Hr = np.zeros((M,N,NN))
	        for l in range(M):
	            perm = np.random.permutation(np.arange(len(rhos)))
	            new_Hr[l] = np.exp(utils.student_logprob(samples,rhos[perm],dfs[perm]))
	        kR+=1
	        Hr += (new_Hr.mean(axis=0) - Hr)/kR
	        v += ((new_Hr-Hr)**2).sum(axis=0)
	        sR = norm.ppf(1 - 0.05) * (v / (kR * M * (kR * M - 1)))**(.5)
	        tol = (sR/Hr.mean(axis=0)).max()
	#         if kR%10==0:
	#             print(tol)
	    t2 = time.time()
	    Hr = np.log(Hr).mean(axis=-1)
	    
	    k+=1
	    HR += (Hr.mean()-HR)/k
	    HRS += (Hrs.mean()-HRS)/k
	    assert len(Hr)==N
	    var_sum += ((Hr-HR)**2).sum() + ((Hrs-HRS)**2).sum()
	    sem = norm.ppf(1 - 0.05) * (var_sum / (k * N * (k * N - 1)))**(.5)
	    print(HRS-HR,sem, t2-t1, kR)
	integral = ((HRS-HR,HRS)/np.log(2))

	for repetition in range(3):
		print(f"Nvar={Nvar}, {repetition+1}/3")

		t1 = time.time()

		y = utils.student_rvs(Nvar,rhos,dfs,1).squeeze()
		y0 = np.zeros_like(y)
		for i in range(y.shape[0]):
		    y0[i] = t.cdf(y[i],df=dfs[i])

		KSG = mg.revised_mi(x.reshape((*x.shape,1)),y0,)

		N = Nvar
		#redirect logging here
		exp_pref='benchmark'
		data_layers = [torch.tensor(y0).float().to(device)]
		copula_layers = []
		for m in range(0,N-1):
		    out_dir = f'{conf.path2outputs}/{exp_pref}/layer{m}'
		    copulas, layer = [], []
		    for n in range(1,N-m):
		        print(m,n+m)
		        samples = data_layers[-1][...,[n,0]]
		        waic, model = bvcopula.infer([bvcopula.GaussianCopula_Likelihood(),
		                                      # bvcopula.GumbelCopula_Likelihood(rotation='0°'),
		                                      bvcopula.GumbelCopula_Likelihood(rotation='180°')],
		                                     train_x,samples,device=device)
		        # (likelihoods, waic) = select_copula.select_with_heuristics(x,samples.cpu().numpy(),device,exp_pref,out_dir,m,n+m,
		        #                                                           train_x=train_x, train_y=samples)
		        # weights_file = f"{out_dir}/model_{exp_pref}_{m}-{m+n}.pth"
		        # model = utils.get_model(weights_file, likelihoods, device) 
		        with torch.no_grad():
		            f = model(train_x).mean
		            copula = model.likelihood.get_copula(f)
		            copulas.append(copula)
		            layer.append(copula.ccdf(samples))
		    data_layers.append(torch.stack(layer,dim=-1))
		    copula_layers.append(copulas)

		vine_trained = CVine(copula_layers,train_x,device=device)

		subvine = vine_trained.create_subvine(torch.arange(0,NSamp,50))
		CopulaGP = subvine.stimMI(s_mc_size=50, r_mc_size=20, sem_tol=0.01, v=True)

		t2 = time.time()
		print(f"Took: {(t2-t1)//60} min")

		res = [[Nvar,NSamp],[HRgS.mean(),CopulaGP[2].item(),-KSG[1]],
				[integral[0],CopulaGP[0].item(),KSG[0],MI.Mixed_KSG(x,y0)],[y0,rhos,dfs]]

		print(res[1])
		print(res[2])

		results_file = f"{home}/MI_dump.pkl"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append(res)

		with open(f"{home}/MI_dump.pkl","wb") as f:
			pkl.dump(results,f)	