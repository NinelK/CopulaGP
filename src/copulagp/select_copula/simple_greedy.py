import torch
import numpy as np
from torch import Tensor
from matplotlib import pyplot as plt
from gpytorch.distributions import MultivariateNormal
import logging
import os

import copulagp.bvcopula as bvcopula
import copulagp.utils as utils
from . import conf
from .importance import important_copulas, reduce_model

def available_elements(current_model):
	'''
	Checks which elements from the list are still available (not yet used in a current_model).
	In other words, returns a complimentary set to the set of the elements already used in the model.
	'''
	if type(current_model) != list:
		current_model = [current_model]
	av_el = []
	for el in conf.elements:
	    available=True
	    for used in current_model:
	        if (used.name==el.name) & (used.rotation==el.rotation):
	            available=False
	    if available:
	    	av_el.append(el)
	return av_el

def add_copula(X: Tensor, Y: Tensor, train_x: Tensor, train_y: Tensor, device: torch.device,
	simple_model: list,	exp_name: str, path_output: str, name_x: str, name_y: str):

	if type(simple_model) != list:
		simple_model = [simple_model]

	available = available_elements(simple_model)
	waics = np.ones(len(available)) * (-float("Inf"))
	files_created = []
	for i, el in enumerate(available): #iterate over absolute indexes of available elements
		likelihoods = [el]+simple_model
		# make file names
		name = f'{exp_name}_{utils.get_copula_name_string(likelihoods)}'
		weights_filename = f'{path_output}/w_{name}.pth'
		#################
		waic = float("Inf")
		try:
			waic, model = bvcopula.infer(likelihoods,train_x,train_y,device=device)
			torch.save(model.gp_model.state_dict(),weights_filename)
			files_created.append(weights_filename)
		except ValueError as error:
			logging.error(error)
			logging.error(utils.get_copula_name_string(likelihoods),' failed')
		finally:
			waics[i] = waic

	best_i = np.argmin(waics)
	best = available[best_i]
	logging.info(f"Best added copula: {best.name} {utils.strrot(best.rotation)} (WAIC = {np.min(waics):.4f})")

	best_likelihoods = [best] + simple_model # order here is extrimely important!!!
	waic = np.min(waics)

	# load the best model to plot
	name = f'{exp_name}_{utils.get_copula_name_string(best_likelihoods)}'
	weights_filename = f'{path_output}/w_{name}.pth'
	model = bvcopula.load_model(weights_filename, best_likelihoods, device)

	# plot the result
	plot_res = f'{path_output}/res_{name}.png'
	utils.Plot_Fit(model, X, Y, name_x, name_y, plot_res, device=device)

	# remove all weights for all models, except for the best one
	for file in files_created[1:]: # never delete independence
		if file!=weights_filename:
			logging.debug(f'Removing {file}')
			os.remove(file)

	return (best_likelihoods,waic)

def select_copula_model(X: Tensor, Y: Tensor, device: torch.device,
	exp_pref: str, path_output: str, name_x: str, name_y: str,
	train_x = None, train_y = None):

	exp_name = f'{exp_pref}_{name_x}-{name_y}'
	log_name = f'{path_output}/log_{device}_{exp_name}.txt'
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	logging.basicConfig(filename=log_name, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

	#convert numpy data to tensors (optionally on GPU)
	if train_x is None:
		train_x = torch.tensor(X).float().to(device=device)
	if train_y is None:
		train_y = torch.tensor(Y).float().to(device=device)
	
	mixtures = [[]]
	waics = [float("inf")]
	num_elements = 0
	while num_elements < conf.max_mix:
		(likelihoods, waic) = add_copula(X,Y,train_x,train_y,device,mixtures[-1],exp_name,path_output,name_x,name_y)
		num_elements = len(likelihoods)
		if (waic > conf.waic_threshold):
			logging.info(f'The variables are independent (waic less than {conf.waic_threshold:.4f}).')	
			mixtures.append([bvcopula.IndependenceCopula_Likelihood()])
			waics.append(0)
			break
		elif (waic <= min(waics)): #(num_elements<3) |
			mixtures.append(likelihoods)
			waics.append(waic)
		else:
			logging.info('The last added copula did not increase the likelihood.')	
			break
		
	best_ind = np.argmin(waics)
	logging.info(f"The best model is {utils.get_copula_name_string(mixtures[best_ind])} with WAIC = {waics[best_ind]:.4f}")

	# load the best model to check, if reduction is needed
	name = f'{exp_name}_{utils.get_copula_name_string(mixtures[best_ind])}'
	weights_filename = f'{path_output}/w_{name}.pth'
	model = bvcopula.load_model(weights_filename, mixtures[best_ind], device)
	#reduce the model
	important = important_copulas(model)
	reduced_likelihoods = reduce_model(mixtures[best_ind],important) 
	if np.any(available_elements(reduced_likelihoods) != available_elements(mixtures[best_ind])):
		logging.info("Model was reduced, getting new WAIC...")
		waic, model = bvcopula.infer(reduced_likelihoods,train_x,train_y,device=device)
		name = f'{exp_name}_{utils.get_copula_name_string(reduced_likelihoods)}'
		weights_filename = f'{path_output}/w_{name}.pth'
		torch.save(model.gp_model.state_dict(),weights_filename)
		print(f"Model reduced to {utils.get_copula_name_string(reduced_likelihoods)}")
		waics[best_ind] = waic.cpu().numpy()
		mixtures[best_ind] = reduced_likelihoods
		# plot the result
		plot_res = f'{path_output}/res_{name}.png'
		utils.Plot_Fit(model, X, Y, name_x, name_y, plot_res, device=device)

	# copy the very best model 
	# if True: #(utils.get_copula_name_string(mixtures[best_ind])!='Independence'):
	# 	...

	print('History:')
	for mix,waic in zip(mixtures[1:],waics[1:]):
		print(f"{utils.get_copula_name_string(mix)} with WAIC = {waic:.4f}")

	return (mixtures[best_ind],waics[best_ind])

