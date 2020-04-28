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

NSamp=10000
device = torch.device('cuda:0')
x = torch.linspace(0.,1.,NSamp).numpy()
test_x = torch.tensor(x).float().to(device=device)

for M in range(3,10):
	for repetition in range(10):
		print(f"M={M}, {repetition+1}/10")
		vine = utils.get_random_vine(M,test_x,max_el=1,device=device)
		print(utils.get_vine_name(vine))
		res = np.zeros((3,2))
		for i in range(3):
		    V = vine.sample()
		    y = V.cpu().numpy()
		    res[i] = mg.revised_mi(x.reshape((*x.shape,1)),y)
		t1 = time.time()
		subvine = vine.create_subvine(torch.arange(0,NSamp,20))
		stimMI = subvine.stimMI(sem_tol=0.01)
		t2 = time.time()
		print(f"Took: {(t2-t1)//60} min")

		results_file = f"{home}/MI_dump.pkl"
		if os.path.exists(results_file):
			with open(results_file,'rb') as f:
				results = pkl.load(f)  
		else:
			results = []

		results.append((vine,(res.mean(axis=0),res.std(axis=0)),stimMI,
						MI.Mixed_KSG(x,y)/np.log(2)))

		with open(f"{home}/MI_dump.pkl","wb") as f:
			pkl.dump(results,f)	