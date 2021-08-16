import time
import pickle as pkl
import numpy as np
import multiprocessing
import os
from torch import device, tensor, load, no_grad

import copulagp.select_copula as select_copula
import copulagp.bvcopula as bvcopula
from copulagp.select_copula import conf as conf_select

def worker(X, Y0, Y1, idxs, layer, gauss=False, light=False, shuffle=False):
	# get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = (int(cpu_name[cpu_name.find('-') + 1:]) - 1)%len(device_list) # ids will be 8 consequent numbers
	device_str = device_list[cpu_id]

	Y = np.stack([Y1,Y0]).T # order!
	n0, n1, n_out = idxs[0] + layer, idxs[1]+layer, idxs[1]-1 # substitute this to get other (not C) vines

	train_x = tensor(X).float().to(device=device(device_str))
	train_y = tensor(Y).float().to(device=device(device_str))

	# print(f'Selecting {n0}-{n1} on {device_str}')
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
			if light:
				(store, waic) = select_copula.select_light(X,Y,device(device_str),exp_pref,log_dir,n0,n1,train_x=train_x,train_y=train_y)
			else:
				(store, waic) = select_copula.select_with_heuristics(X,Y,device(device_str),exp_pref,log_dir,n0,n1,train_x=train_x,train_y=train_y)
			model = store.model_init(device(device_str))
			# (likelihoods, waic) = select_copula.select_copula_model(X,Y,device(device_str),exp_pref,log_dir,layer,n+layer)
		t_end = time.time()
		# print(f'Selection took {int((t_end-t_start)/60)} min')
	except RuntimeError as error:
		print(error)
		# logging.error(error, exc_info=True)
		return -1
	finally:
		print(f"{n0}-{n1} {store.name_string} {waic:.4} took {int((t_end-t_start)/60)} min")
		# save textual info into model list
		if log_dir!=None:
			with open(log_dir+'_model_list.txt','a') as f:
				f.write(f"{n0}-{n1} {store.name_string}\t{waic:.4f}\t{int(t_end-t_start)} sec\n")

		if store.name_string!='Independence':
			model.gp_model.eval()
			copula = model.marginalize(train_x) # marginalize the GP
			y = copula.ccdf(train_y).cpu().numpy()
		else:
			y = Y1

		return (store, waic, y)

def train_next_tree(X: np.ndarray, Y: np.ndarray, 
		    layer: int, devices: list, gauss=False, light=False, shuffle=False, path_logs=lambda x,y: None,
	exp = ''):
	'''
	Trains one vine copula tree

	Parameters
	----------
	X : np.ndarray
		Conditioning variable
	Y : np.ndarray
		Collection of data variables
	exp : str (Default = '')
		Name of the experiment.
		If empty: do not save checkpoints
		or logs.
	layer : int 
		Layer (a.k.a. tree) number
	device_list : List[str]
		A list of devices to be used for
		training (in parallel)
	gauss : bool (Default = False)
		A flag that turns off model selection
		and only trains gaussian copula models

	Returns
	-------
	to_save : dict
		Dictionary with keys={'models','waics'}
	'''
	for dev in devices:
		assert (dev=='cpu') or (dev[:-2]=='cuda')

	global exp_pref, log_dir
	exp_pref = exp
	if gauss:
		exp_pref += '_g'

	global device_list
	device_list = devices

	if exp!='':
		log_dir = path_logs(exp_pref, layer)
		if(log_dir is not None):
			try:
				os.makedirs(log_dir)
			except FileExistsError as error:
				print(f"Error:{error}")
	else:
		log_dir = None

	NN = Y.shape[-1]-1

	results = np.empty(NN,dtype=object)
	pool = multiprocessing.Pool(len(device_list))

	for i in np.arange(1,NN+1): 
		results[i-1] = pool.apply_async(worker, (X, Y[:,0], Y[:,i], [0,i],  layer, gauss, light, shuffle))

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
