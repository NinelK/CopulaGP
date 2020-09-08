import time
import pickle as pkl
import numpy as np
import sys
import multiprocessing
import os
from torch import device
from . import conf
sys.path.insert(0, conf.path2code)

import utils
import select_copula
import train

gpu_id_list = range(4)
unique_id_list = np.random.randint(0,10000,len(gpu_id_list)) #TODO: make truely unique
#[i//2 for i in range(8*2)]  # 2 workers on each GPU

def worker(X, Y0, Ys, idxs, NN, progress, exp_pref, layer):
	# get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = (int(cpu_name[cpu_name.find('-') + 1:]) - 1)%len(gpu_id_list) # ids will be 8 consequent numbers
	gpu_id = gpu_id_list[cpu_id]

	out_dir = f'{conf.path2outputs}/{exp_pref}/layer{layer}'
	device_str = f'cuda:{gpu_id}'

	print(f'Start a new batch ({progress}) on {device_str}')

	unique_id = unique_id_list[cpu_id]

	for n,Y1 in zip(idxs,Ys.T):

		Y = np.stack([Y1,Y0]).T # order!

		print(f'Selecting {layer}-{n+layer} on {device_str}')
		try:
			t_start = time.time()
			# (likelihoods, waic) = select_copula.select_copula_model(X,Y,device(device_str),exp_pref,out_dir,layer,n+layer)
			(likelihoods, waic) = select_copula.select_with_heuristics(X,Y,device(device_str),exp_pref,out_dir,layer,n+layer)
			t_end = time.time()
			print(f'Selection took {int((t_end-t_start)/60)} min')
		except RuntimeError as error:
			print(error)
		finally:
			if NN!=-1:
				print(utils.get_copula_name_string(likelihoods),waic)
				# save textual info into model list
				with open(out_dir+'_model_list.txt','a') as f:
					f.write(f"{layer}-{n+layer} {utils.get_copula_name_string(likelihoods)}\t{waic:.4f}\t{int(t_end-t_start)} sec\n")
				
				# save the layer
				results_file = f"{out_dir}_{unique_id}_models.pkl"
				if os.path.exists(results_file):
					with open(results_file,'rb') as f:
						results = pkl.load(f)  
				else:
					results = np.empty(NN,dtype=object)

				assert (results[n-1]==None)
				results[n-1] = [likelihoods,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)]

				with open(results_file,'wb') as f:
					pkl.dump(results,f)   
			else:
				results_file = f"{out_dir}/pair_model.pkl"
				with open(results_file,'wb') as f:
					results = [likelihoods,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)]
					pkl.dump(results,f)   

	return 0

def train_next_layer(exp_pref, layer, batch = 1):

	if layer==0:
		try:
			os.mkdir(f'{conf.path2outputs}/{exp_pref}')
		except FileExistsError as error:
			print(f"Error:{error}")

	try:
		os.mkdir(f'{conf.path2outputs}/{exp_pref}/layer{layer}')
	except FileExistsError as error:
		print(f"Error:{error}")

	pool = multiprocessing.Pool(len(gpu_id_list))

	X,Y = utils.standard_loader(f"{conf.path2data}/{exp_pref}/{exp_pref}_layer{layer}.pkl")
	#Y = Y0[...,1:]
	#print(Y0.shape,Y.shape)
	NN = Y.shape[-1]-1

	# batch = int(np.ceil(NN/len(gpu_id_list)/repeats))

	print(f"Batch size: {batch}")

	list_idx = np.arange(1,NN+1)
	resid = len(list_idx)%batch
	if resid!=0:
		list_idx = np.concatenate([list_idx,np.zeros(batch-resid)]).astype('int')
	batches = np.reshape(list_idx,(batch,-1)).T

	for i,b in enumerate(batches):
		res = pool.apply_async(worker, (X, Y[:,0], Y[:,b[b!=0]], b[b!=0], NN, f"{i+1}/{len(batches)}", exp_pref, layer, ))

	# for 3plets
	if layer == 1:
		X,Y = utils.standard_loader(f"{conf.path2data}/{exp_pref}/{exp_pref}_layer0.pkl")
		res = pool.apply_async(worker, (X, Y[:,2], Y[:,1].reshape(-1,1), [-1], -1, f"{i+1}/{len(batches)}", exp_pref, layer, ))

	pool.close()
	pool.join()  # block at this line until all processes are done
	print(f"Layer {layer} completed")
