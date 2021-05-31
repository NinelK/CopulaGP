import torch
import numpy as np
import time
from torch import Tensor
from matplotlib import pyplot as plt
import logging
from gpytorch.mlls import VariationalELBO
from gpytorch.settings import num_likelihood_samples
import gc

from copulagp.utils import get_copula_name_string
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

def infer(bvcopulas, train_x: Tensor, train_y: Tensor, device: torch.device,
			output_loss=None, grid_size=None, prior_rbf_length=0.5):

	if device!=torch.device('cpu'):
		with torch.cuda.device(device):
			torch.cuda.empty_cache()

	logging.info(f'Trying {get_copula_name_string(bvcopulas)}')

	# define the model (optionally on GPU)
	from .models import Pair_CopulaGP
	model = Pair_CopulaGP(bvcopulas,device=device,grid_size=grid_size,prior_rbf_length=prior_rbf_length)

	optimizer = torch.optim.Adam([
	    {'params': model.gp_model.mean_module.parameters()},
	    {'params': model.gp_model.variational_strategy.parameters()},
	    {'params': model.gp_model.covar_module.parameters(), 'lr': conf.hyper_lr}, #hyperparameters
	], lr=conf.base_lr)

	# train the model

	mll = VariationalELBO(model.likelihood, model.gp_model,
                            num_data=train_y.size(0))

	losses, rbf, means = [], [], []

	nans_detected = 0
	WAIC = -1 #assume that the model will train well
	
	def train(train_x, train_y, num_iter=conf.max_num_iter):
	    model.gp_model.train()
	    model.likelihood.train()

	    loss_gpu = torch.zeros(num_iter,device=device)

	    p = torch.zeros(1,device=device)
	    nans = torch.zeros(1,device=device)
	    for i in range(num_iter):
	        optimizer.zero_grad()
	        output = model.gp_model(train_x)
	        
	        with num_likelihood_samples(30):
	        	loss = -mll(output, train_y)  
	 
	        if i>2*conf.loss_av: 
	            p += (loss_gpu[i-conf.loss_av:i].mean() - loss_gpu[i-2*conf.loss_av:i-conf.loss_av].mean()).abs()
	        loss_gpu[i] = loss.detach()

	        if not (i + 1) % conf.iter_print:

	            losses.append(loss.detach().cpu().numpy())
	            rbf.append(model.gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze())
	            means.append(model.gp_model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.detach().cpu().numpy())
	            
	            mean_p = p/conf.loss_av/2
	            av_loss = loss_gpu[i-2*conf.loss_av:i].mean().abs().cpu()
                # print(f"{i}: {mean_p}")

	            if (mean_p < conf.loss_tol2check_waic):
	                WAIC = model.likelihood.WAIC(model.gp_model(train_x),train_y)
	                if (WAIC > conf.waic_tol):
	                    logging.debug("Training does not look promissing!")
	                    break	

	            if (mean_p < conf.a_loss_tol) or (mean_p/av_loss < conf.r_loss_tol):
	                a = 'Absolute!' if (mean_p < conf.a_loss_tol) else 'Relative!'
	                logging.debug(f"Converged in {i+1} steps! ({a})")
	                break
	            p = 0.

	            for param_group in optimizer.param_groups:
	            	param_group['lr'] = param_group['lr']*conf.decrease_lr

	        # The actual optimization step
	        loss.backward()
	        covar_grad = model.gp_model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.grad
	        # strict
	        # assert torch.all(covar_grad==covar_grad)
	        #light
	        if torch.any(covar_grad!=covar_grad):
	            for n, par in model.gp_model.named_parameters():
	                grad = par.grad.data
	                if torch.any(grad!=grad):
	                    # print(f'NaN grad in {n}')
	                    nans_detected = 1
	                # nans+=torch.sum(grad!=grad)
	                if torch.any(grad.abs()==float('inf')):
	                    logging.warning("Grad inf... fixing...")
	                    grad = torch.clamp(grad,-1.,1.)
	                grad[grad!=grad] = 0.0
	                par.grad.data = grad
	        optimizer.step()

	t1 = time.time()

	if (len(bvcopulas)!=1) or (bvcopulas[0].name!='Independence'):
		train(train_x,train_y)

	if nans_detected==1:
		logging.warning('NaNs were detected in gradients.')

	if output_loss is not None:
		assert isinstance(output_loss, str)
		plot_loss(output_loss, losses, rbf, means)

	if (WAIC < 0): 
	# if model got to the point where it was better than independence: recalculate final WAIC
		WAIC = model.likelihood.WAIC(model.gp_model(train_x),train_y)

	t2 = time.time()
	logging.info(f'WAIC={WAIC:.4f}, took {int(t2-t1)} sec')

	if device!=torch.device('cpu'):
		with torch.cuda.device(device):
			torch.cuda.empty_cache()

	gc.collect()

	return WAIC, model

def load_model(filename, bvcopulas, device: torch.device):

	logging.info(f'Loading {get_copula_name_string(bvcopulas)}')

	# define the model (optionally on GPU)
	from .models import Pair_CopulaGP
	model = Pair_CopulaGP(bvcopulas,device=device)

	model.gp_model.load_state_dict(torch.load(filename, map_location=device))
	model.gp_model.eval()

	return model
