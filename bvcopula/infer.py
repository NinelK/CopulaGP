import torch
import numpy as np
import time
from torch import Tensor
from matplotlib import pyplot as plt
import logging

import bvcopula
import utils
from . import conf

def plot_loss(filename, losses, rbf, means):
	# prot loss function and kernel length
	fig, (loss, kern, mean) = plt.subplots(1,3,figsize=(16,2))
	loss.plot(losses)
	loss.set_xlabel("Epoch #")
	loss.set_ylabel("Loss")
	marg = (np.max(losses) - np.min(losses))*0.1
	loss.set_ylim(np.min(losses)-marg,
	              np.max(losses)+marg)
	rbf=np.array(rbf).squeeze()
	kern.plot(rbf)
	kern.set_xlabel("Epoch #")
	kern.set_ylabel("Kernel scale parameter")
	mean.plot([np.mean(x,axis=1) for x in means])
	mean.set_xlabel("Epoch #")
	mean.set_ylabel("Mean f")
	fig.savefig(filename)
	plt.close()

def _get_theta_sharing(likelihoods, theta_sharing):
	if theta_sharing is not None:
		theta_sharing = theta_sharing
		num_fs = len(likelihoods)+thetas_sharing.max().numpy() # indep_thetas + num_copulas - 1
	else:
		theta_sharing = torch.arange(0,len(likelihoods)).long()
		num_fs = 2*len(likelihoods)-1
	return theta_sharing, num_fs

def _grid_size(num_copulas):
	if num_copulas<8:
		grid_size = conf.grid_size
	else:
		grid_size = int(conf.grid_size/int(np.log(num_copulas)/np.log(2)))
		print(grid_size)
	return grid_size

def infer(likelihoods, train_x: Tensor, train_y: Tensor, device: torch.device,
			theta_sharing=None,
			output_loss = None):

	theta_sharing, num_fs = _get_theta_sharing(likelihoods, theta_sharing)

	if device!=torch.device('cpu'):
		with torch.cuda.device(device):
			torch.cuda.empty_cache()

	logging.info('Trying {}'.format(utils.get_copula_name_string(likelihoods)))

	# define the model (optionally on GPU)
	model = bvcopula.Mixed_GPInferenceModel(
			bvcopula.MixtureCopula_Likelihood(likelihoods, 
								theta_sharing=theta_sharing), 
                            num_fs,  
                            prior_rbf_length=0.5, 
                            grid_size=_grid_size(len(likelihoods))).to(device=device).float()

	optimizer = torch.optim.Adam([
	    {'params': model.mean_module.parameters()},
	    {'params': model.variational_strategy.parameters()},
	    {'params': model.covar_module.parameters(), 'lr': conf.hyper_lr}, #hyperparameters
	], lr=conf.base_lr)

	# train the model

	mll = utils.VariationalELBO(model.likelihood, model, torch.ones_like(train_x.squeeze()), 
                            num_data=train_y.size(0), particles=torch.Size([0]), combine_terms=True)

	losses = torch.zeros(conf.max_num_iter, device=device)
	rbf = torch.zeros(conf.max_num_iter, device=device)
	means = torch.zeros(conf.max_num_iter, device=device)
	nans_detected = 0
	WAIC = -1 #assume that the model will train well
	
	def train(train_x, train_y, num_iter=conf.max_num_iter):
	    model.train()

	    p = 0.
	    nans = 0
	    for i in range(num_iter):
	        optimizer.zero_grad()
	        output = model(train_x)
	        
	        loss = -mll(output, train_y)  
	 
	        losses[i] = loss.detach()
	        #rbf[i] = model.covar_module.base_kernel.lengthscale.detach()
	        #means[i] = model.variational_strategy.variational_distribution.variational_mean\
	        #		.detach()

	        if len(losses)>100: 
	            p += torch.abs(torch.mean(losses[i-50:i+1]) - torch.mean(losses[i-100:i-50]))

	        if not (i + 1) % conf.iter_print:
	            
	            mean_p = p/100

	            if (0 < mean_p < conf.loss_tol2check_waic):
	                WAIC = model.likelihood.WAIC(model(train_x),train_y)
	                if (WAIC > conf.waic_tol):
	                    logging.debug("Training does not look promissing!")
	                    break	

	            if (0 < mean_p < conf.loss_tol):
	                logging.debug("Converged in {} steps!".format(i+1))
	                break
	            p = 0.

	        # The actual optimization step
	        loss.backward()
	        covar_grad = model.variational_strategy.variational_distribution.chol_variational_covar.grad
	        # strict
	        # assert torch.all(covar_grad==covar_grad)
	        #light
	        if torch.any(covar_grad!=covar_grad):
	            for n, par in model.named_parameters():
	                grad = par.grad.data
	                if torch.nonzero(grad!=grad).shape[0]!=0:
	                    #print('NaN grad in {}'.format(n))
	                    nans_detected = 1
	                nans+=torch.nonzero(grad!=grad).shape[0]
	                if torch.any(grad.abs()==float('inf')):
	                    logging.warning("Grad inf... fixing...")
	                    grad = torch.clamp(grad,-1.,1.)
	                grad[grad!=grad] = 0.0
	                par.grad.data = grad
	        optimizer.step()

	t1 = time.time()

	if (len(likelihoods)!=1) | (likelihoods[0].name!='Independence'):
		train(train_x,train_y)

	if nans_detected==1:
		logging.warning('NaNs were detected in gradients.')

	if output_loss is not None:
		assert isinstance(output_loss, str)
		plot_loss(output_loss, losses.cpu().numpy(), rbf.cpu().numpy(), means.cpu().numpy())

	if (WAIC < 0): 
	# if model got to the point where it was better than independence: recalculate final WAIC
		WAIC = model.likelihood.WAIC(model(train_x),train_y)

	t2 = time.time()
	logging.info('WAIC={:.4f}, took {} sec'.format(WAIC,int(t2-t1)))

	if device!=torch.device('cpu'):
		with torch.cuda.device(device):
			torch.cuda.empty_cache()

	return WAIC, model

def load_model(filename, likelihoods, device: torch.device, 
	theta_sharing=None):

	theta_sharing, num_fs = _get_theta_sharing(likelihoods, theta_sharing)

	logging.info('Loading {}'.format(utils.get_copula_name_string(likelihoods)))

	# define the model (optionally on GPU)
	model = bvcopula.Mixed_GPInferenceModel(
			bvcopula.MixtureCopula_Likelihood(likelihoods, 
								theta_sharing=theta_sharing), 
                            num_fs,  
                            prior_rbf_length=0.5, 
                            grid_size=_grid_size(len(likelihoods))).to(device=device)

	model.load_state_dict(torch.load(filename, map_location=device))
	model.eval()

	return model
