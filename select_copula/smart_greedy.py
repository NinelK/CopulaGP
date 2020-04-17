import torch
import numpy as np
from torch import Tensor
from matplotlib import pyplot as plt
from gpytorch.distributions import MultivariateNormal
import logging
import os

import bvcopula
import utils
from . import conf
from .importance import important_copulas

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


# def important_copulas(model: bvcopula.Mixed_GPInferenceModel, train_x: Tensor) -> Tensor:

# 	if len(model.likelihood.likelihoods)<2:
# 		return torch.tensor([True])
# 	else:
		    
# 		model.eval()
# 		with torch.no_grad():
# 		    output = model(train_x)
# 		gplink = model.likelihood.gplink_function
# 		_, control = gplink(output.mean)
# 		_, mixes = gplink(output.rsample(torch.Size([1000])))
# 		lowest_mixes = torch.mean(mixes,dim=1) - torch.std(mixes,dim=1)
# 		return (torch.mean(lowest_mixes,dim=1)>0.1).type(torch.bool)

def reduce_model(likelihoods: list, importance: Tensor) -> list:
    assert len(likelihoods)==len(importance)
    reduced_model = []
    for el, imp in zip(likelihoods,importance):
        if imp:
            reduced_model.append(el)
    return reduced_model

def add_copula(X: Tensor, Y: Tensor, train_x: Tensor, train_y: Tensor, device: torch.device,
	simple_model: bvcopula.Mixed_GPInferenceModel,
	exp_name: str, path_output: str, name_x: str, name_y: str,
	reduce=True):

	if type(simple_model) != list:
		simple_model = [simple_model]

	available = available_elements(simple_model)
	waics = np.ones(len(available)) * (-float("Inf"))
	all_important_copulas = [[] for i in range(len(available))]
	files_created = []
	for i, el in enumerate(available): #iterate over absolute indexes of available elements
		likelihoods = [el]+simple_model
		# make file names
		name = '{}_{}'.format(exp_name,utils.get_copula_name_string(likelihoods))
		#plot_loss = '{}/loss_{}.png'.format(path_output,name)
		#plot_res = '{}/res_{}.png'.format(path_output,name)
		weights_filename = '{}/w_{}.pth'.format(path_output,name)
		#################
		waic = float("Inf")
		try:
			waic, model = bvcopula.infer(likelihoods,train_x,train_y,device=device)#,output_loss=plot_loss)
			#plot_fit(model, X, Y, name_x, name_y, plot_res, device=device)
			torch.save(model.state_dict(),weights_filename)
			files_created.append(weights_filename)
			important = important_copulas(model,device)
			if torch.any(important==False):
				logging.info('Only {} were important'.format(utils.get_copula_name_string(reduce_model(likelihoods,important))))
		except ValueError as error:
			logging.error(error)
			logging.error(utils.get_copula_name_string(likelihoods),' failed')
		finally:
			waics[i] = waic
			all_important_copulas[i] = important

	best_i = np.argmin(waics)
	best = available[best_i]
	print("Best added copula: {} {} (WAIC = {:.4f})".format(best.name,utils.strrot(best.rotation),np.min(waics)))
	logging.info("Best added copula: {} {} (WAIC = {:.4f})".format(best.name,utils.strrot(best.rotation),np.min(waics)))

	best_likelihoods = [best] + simple_model # order here is extrimely important!!!
	waic = np.min(waics)

	if reduce:
		#try and reduce the best model
		reduced_likelihoods = reduce_model(best_likelihoods,all_important_copulas[best_i]) 
		name = '{}_{}'.format(exp_name,utils.get_copula_name_string(reduced_likelihoods))
		weights_filename = '{}/w_{}.pth'.format(path_output,name)
		if np.any(available_elements(reduced_likelihoods) != available_elements(best_likelihoods)) & \
			np.any(available_elements(reduced_likelihoods) != available_elements(simple_model)):
			logging.info("Model was reduced, getting new WAIC...")
			waic, model = bvcopula.infer(reduced_likelihoods,train_x,train_y,device=device)
			torch.save(model.state_dict(),weights_filename)
			print("Model reduced to {}".format(utils.get_copula_name_string(reduced_likelihoods)))
			waic = waic.cpu().numpy()
			best_likelihoods = reduced_likelihoods
		else:
			model = bvcopula.load_model(weights_filename, reduced_likelihoods, device)
	else:
		# load the best model to plot
		name = '{}_{}'.format(exp_name,utils.get_copula_name_string(best_likelihoods))
		weights_filename = '{}/w_{}.pth'.format(path_output,name)
		model = bvcopula.load_model(weights_filename, best_likelihoods, device)

	# plot the result
	plot_res = '{}/res_{}.png'.format(path_output,name)
	utils.Plot_Fit(model, X, Y, name_x, name_y, plot_res, device=device)

	# remove all weights for all models, except for the best one
	for file in files_created:
		if file!=weights_filename:
			logging.debug('Removing {}'.format(file))
			os.remove(file)

	return (best_likelihoods,waic)

def _check_history(likelihoods,history):
	res = True #assume that likelihood is new
	for h in history:
		if available_elements(likelihoods) == available_elements(h):
			res = False # likelihood found in history
	return res

def select_copula_model(X: Tensor, Y: Tensor, device: torch.device,
	exp_pref: str, path_output: str, name_x: str, name_y: str,
	train_x = None, train_y = None):

	exp_name = '{}_{}-{}'.format(exp_pref,name_x,name_y)
	log_name = '{}/log_{}_{}.txt'.format(path_output,device,exp_name)
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
		if _check_history(likelihoods,mixtures): #if nothing changed since last iteration
			if (waic > conf.waic_threshold):
				logging.info('The variables are independent (waic less than {:.4f}).'.format(conf.waic_threshold))	
				mixtures.append([bvcopula.IndependenceCopula_Likelihood()])
				waics.append(0)
				break
			elif (num_elements<3) | (waic <= min(waics)):
				mixtures.append(likelihoods)
				waics.append(waic)
			else:
				logging.info('The last added copula did not increase the likelihood.')	
				break
		else:
			logging.info('The last added copula was useless in a mixture.')
			mixtures.append(likelihoods)
			waics.append(waic)
			break

	best_ind = np.argmin(waics)
	print("The best model is {} with WAIC = {:.4f}".format(utils.get_copula_name_string(mixtures[best_ind]),waics[best_ind]))
	logging.info("The best model is {} with WAIC = {:.4f}".format(utils.get_copula_name_string(mixtures[best_ind]),waics[best_ind]))

	# copy the very best model 
	if (utils.get_copula_name_string(mixtures[best_ind])!='Independence'):
		name = '{}_{}'.format(exp_name,utils.get_copula_name_string(mixtures[best_ind]))
		source = '{}/w_{}.pth'.format(path_output,name)
		target = '{}/model_{}.pth'.format(path_output,exp_name)
		os.popen('cp {} {}'.format(source,target)) 
		source = '{}/res_{}.png'.format(path_output,name)
		target = '{}/best_{}.png'.format(path_output,exp_name)
		os.popen('cp {} {}'.format(source,target))

	print('History:')
	for mix,waic in zip(mixtures[1:],waics[1:]):
		print("{} with WAIC = {:.4f}".format(utils.get_copula_name_string(mix),waic))

	return (mixtures[best_ind],waics[best_ind])

