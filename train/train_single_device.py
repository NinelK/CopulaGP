import time
import pickle as pkl
import numpy as np
import os
from . import conf
import sys
sys.path.insert(0, conf.path2code)

import utils
import select_copula
import train

def train(X, Y0, Ys, idxs, NN, exp_pref, layer, device):

	out_dir = f'{conf.path2outputs}/{exp_pref}/layer{layer}'
	results = np.empty(NN,dtype=object)

	for n,Y1 in zip(idxs,Ys.T):

		Y = np.stack([Y1,Y0]).T # order!

		print(f'Selecting {layer}-{n+layer} on {device}')
		try:
			t_start = time.time()
			# (likelihoods, waic) = select_copula.select_copula_model(X,Y,device,exp_pref,out_dir,layer,n+layer)
			(likelihoods, waic) = select_copula.select_with_heuristics(X,Y,device,exp_pref,out_dir,layer,n+layer)
			t_end = time.time()
			print('Selection took {} min'.format(int((t_end-t_start)/60)))
		except RuntimeError as error:
			print(error)
		finally:
			print(utils.get_copula_name_string(likelihoods),waic)

			assert (results[n-1]==None)
			results[n-1] = [likelihoods,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)]

	return results

def train_next_layer(exp_pref, layer, batch = 1, device='cpu'):

	if layer==0:
		try:
			os.mkdir(f'{conf.path2outputs}/{exp_pref}')
		except FileExistsError as error:
			print(error)

	try:
		os.mkdir(f'{conf.path2outputs}/{exp_pref}/layer{layer}')
	except FileExistsError as error:
		print(error)

	X,Y = utils.standard_loader(f"{conf.path2data}/{exp_pref}_layer{layer}.pkl")
	NN = Y.shape[-1]-1

	results = train(X, Y[:,0], Y[:,1:], np.arange(1,NN+1), NN, exp_pref, layer, device)

	print(f"Layer {layer} completed")

	return results
