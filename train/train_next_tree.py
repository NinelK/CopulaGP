import time
import pickle as pkl
import numpy as np
import sys
import multiprocessing
import os
from torch import device, tensor, load
from . import conf
sys.path.insert(0, conf.path2code)

import select_copula
import train
import bvcopula
from select_copula import conf as conf_select

def worker(X, Y0, Y1, idxs, layer, gauss=False):
	# get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = (int(cpu_name[cpu_name.find('-') + 1:]) - 1)%len(device_list) # ids will be 8 consequent numbers
	device_str = device_list[cpu_id]

	Y = np.stack([Y1,Y0]).T # order!
	n0, n1, n_out = idxs[0] + layer, idxs[1]+layer, idxs[1]-1 # substitute this to get other (not C) vines

	train_x = tensor(X).float().to(device=device(device_str))
	train_y = tensor(Y).float().to(device=device(device_str))

	print(f'Selecting {n0}-{n1} on {device_str}')
	try:
		t_start = time.time()
		if gauss:
			gauss = [bvcopula.GaussianCopula_Likelihood()]
			waic, model = bvcopula.infer(gauss,train_x,train_y,device=device(device_str)) 
			if waic>conf_select.waic_threshold:
				store = bvcopula.Pair_CopulaGP_data([['Independence',None]], None)
			else:
				store = model.cpu().serialize()
		else:
			(store, waic) = select_copula.select_light(X,Y,device(device_str),
							exp_pref,out_dir,n0,n1,train_x=train_x,train_y=train_y)
			model = store.model_init(device(device_str))
			# (likelihoods, waic) = select_copula.select_copula_model(X,Y,device(device_str),exp_pref,out_dir,layer,n+layer)
		t_end = time.time()
		print(f'Selection took {int((t_end-t_start)/60)} min')
	except RuntimeError as error:
		print(error)
		# logging.error(error, exc_info=True)
		return -1
	finally:
		print(f"{n0}-{n1}",store.name_string,waic)
		# save textual info into model list
		with open(out_dir+'_model_list.txt','a') as f:
			f.write(f"{n0}-{n1} {store.name_string}\t{waic:.4f}\t{int(t_end-t_start)} sec\n")

		if store.name_string!='Independence':
			model.gp_model.eval()
			copula = model.marginalize(train_x) # marginalize the GP
			y = copula.ccdf(train_y).cpu().numpy()
		else:
			y = Y1

		return (store, waic, y)

def train_next_tree(X: np.ndarray, Y: np.ndarray, 
	exp: str, layer: int, devices: list, gauss=False):

	for dev in devices:
		assert (dev=='cpu') or (dev[:-2]=='cuda')

	global exp_pref, out_dir
	exp_pref = exp
	out_dir = f'{conf.path2outputs}/logs/layer{layer}'

	global device_list
	device_list = devices

	if layer==0:
		try:
			os.mkdir(f'{conf.path2outputs}/logs')
		except FileExistsError as error:
			print(f"Error:{error}")

	try:
		os.mkdir(out_dir)
	except FileExistsError as error:
		print(f"Error:{error}")

	NN = Y.shape[-1]-1

	results = np.empty(NN,dtype=object)
	pool = multiprocessing.Pool(len(device_list))

	for i in np.arange(1,NN+1): 
		results[i-1] = pool.apply_async(worker, (X, Y[:,0], Y[:,i], [0,i],  layer, gauss, ))

	pool.close()
	pool.join()  # block at this line until all processes are done
	print(f"Layer {layer} completed")

	models, waics, Y_next = [], [], []
	for result in results:
		m, w, y = result.get()
		models.append(m)
		waics.append(w)
		Y_next.append(y)

	Y_next = np.array(Y_next).T

	return models, waics, Y_next
