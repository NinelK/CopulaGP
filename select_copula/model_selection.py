import time
import torch
from gpytorch.settings import num_likelihood_samples
import numpy as np
from torch import Tensor
from matplotlib import pyplot as plt
from gpytorch.distributions import MultivariateNormal

import bvcopula
import utils

def strrot(rotation):
	if rotation is not None:
		return rotation
	else:
		return ''

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

def infer(likelihoods, X, Y, device: torch.device,
			theta_sharing=None, 
			NUM_ITER=2000, iter_print = 100,
			STOP_LOSS = 0.001,
			output_loss = None):

	if theta_sharing is not None:
		theta_sharing = theta_sharing
		num_fs = len(likelihoods)+thetas_sharing.max().numpy() # indep_thetas + num_copulas - 1
	else:
		theta_sharing = torch.arange(0,len(likelihoods)).long()
		num_fs = 2*len(likelihoods)-1

	print('Trying ',*[lik.name+' '+strrot(lik.rotation) for lik in likelihoods])

	#convert numpy data to tensors (optionally on GPU)
	train_x = torch.tensor(X).float().cuda(device=device)
	train_y = torch.tensor(Y).float().cuda(device=device)

	# define the model (optionally on GPU)
	grid_size = 128
	model = bvcopula.Mixed_GPInferenceModel(
			bvcopula.MixtureCopula_Likelihood(likelihoods, 
								theta_sharing=theta_sharing), 
                            num_fs,  
                            prior_rbf_length=0.5, 
                            grid_size=grid_size).cuda(device=device)

	optimizer = torch.optim.Adam([
	    {'params': model.mean_module.parameters()},
	    {'params': model.variational_strategy.parameters()},
	    {'params': model.covar_module.parameters(), 'lr': .001}, #hyperparameters
	], lr=1e-2)

	# train the model

	mll = utils.VariationalELBO(model.likelihood, model, torch.ones_like(train_x.squeeze()), 
                            num_data=train_y.size(0), particles=torch.Size([0]), combine_terms=True)

	losses, rbf, means = [], [], []

	nans_detected = 0

	def train(train_x, train_y, num_iter=NUM_ITER):
	    model.train()

	    p = 0.
	    nans = 0
	    for i in range(num_iter):
	        optimizer.zero_grad()
	        output = model(train_x)
	        
	        loss = -mll(output, train_y)  
	        if len(losses)>100: 
	            p += np.abs(np.mean(losses[-50:]) - np.mean(losses[-100:-50]))
	        losses.append(loss.detach().cpu().numpy())
	        rbf.append(model.covar_module.base_kernel.lengthscale.detach().cpu().numpy())
	        means.append(model.variational_strategy.variational_distribution.variational_mean\
	        		.detach().cpu().numpy())

	        if not (i + 1) % iter_print:
	            
	            mean_p = p/100

	            if (0 < mean_p < STOP_LOSS/(len(likelihoods)*1.0)):
	                print("Converged in {} steps!".format(i+1))
	                break
	            p = 0.

	        # The actual optimization step
	        loss.backward()
	        covar_grad = model.variational_strategy.variational_distribution.chol_variational_covar.grad
	#         # strict
	#         assert torch.all(covar_grad==covar_grad)
	        #light
	        if torch.any(covar_grad!=covar_grad):
	            for n, par in model.named_parameters():
	                grad = par.grad.data
	                if torch.nonzero(grad!=grad).shape[0]!=0:
	                    #print('NaN grad in {}'.format(n))
	                    nans_detected = 1
	                nans+=torch.nonzero(grad!=grad).shape[0]
	                if torch.any(grad.abs()==float('inf')):
	                    print("Grad inf... fixing...")
	                    grad = torch.clamp(grad,-1.,1.)
	                grad[grad!=grad] = 0.0
	                par.grad.data = grad
	        optimizer.step()

	if nans_detected==1:
		print('NaNs were detected in gradients.')
	       
	t1 = time.time()
	train(train_x,train_y)
	t2 = time.time()
	print(*[lik.name for lik in likelihoods],' took ',int(t2-t1),' sec')

	model.eval()

	if output_loss is not None:
		assert isinstance(output_loss, str)
		plot_loss(output_loss, losses, rbf, means)

	WAIC = model.likelihood.WAIC(model(train_x),train_y)

	print('WAIC={:.3}'.format(WAIC))

	return WAIC, model

def generate_test_samples(model: bvcopula.Mixed_GPInferenceModel, test_x: Tensor) -> Tensor:
	
	with torch.no_grad():
	    output = model(test_x)

	#generate some samples
	model.eval()
	with torch.no_grad(), num_likelihood_samples(1):
	    gplink = model.likelihood.gplink_function
	    likelihoods = model.likelihood.likelihoods
	    copulas = [lik.copula for lik in likelihoods]
	    rotations = [lik.rotation for lik in likelihoods]
	    thetas, mixes = gplink(output.mean)
	    test_y = model.likelihood.copula(thetas,mixes,
	    			copulas, rotations=rotations,
	    			theta_sharing=model.likelihood.theta_sharing).rsample()
	    Y_sim = test_y.cpu().detach().numpy()

	return Y_sim

def important_copulas(model: bvcopula.Mixed_GPInferenceModel, output: MultivariateNormal,
						C_THR = 0.1):

	gplink = model.likelihood.gplink_function
	_, mixes = gplink(output.mean)

	important_copulas = np.zeros(len(likelihoods)).astype("bool")

	for i, t in enumerate(mixes[:]):
	    F_mean = t.detach().cpu().numpy()
	    if np.any(np.abs(F_mean)>C_THR):
	    	important_copulas[i] = True

	return important_copulas 

def plot_pearson(X: Tensor, Y: Tensor):
	from scipy.stats import pearsonr

	X = X.squeeze()
	assert np.isclose(X.max(),1.0)
	assert np.isclose(X.min(),0.0)
	N = int(160/2.5)
	x = np.linspace(0,1,N)
	p = np.empty(N)

	for b in range(N):
	    dat = Y[(X>b*(1./N)) & (X<(b+1)*(1./N))]
	    if len(dat)>1:
	        p[b] = pearsonr(*dat.T)[0]
	    
	p = np.convolve(np.array(p), np.ones((4,))/4, mode='valid')    

	return np.stack([x[2:-1],p])

def plot_fit(model: bvcopula.Mixed_GPInferenceModel, X: Tensor, Y: Tensor,
			name_x: str, name_y: str, filename: str,
			device: torch.device):
	# visualize the result
	fig = plt.figure(figsize=(10, 6))

	top_axes = (fig.add_axes([0.08,0.54,0.4,0.4]),fig.add_axes([0.58,0.54,0.4,0.4]))
	bottom_axes = np.array([fig.add_axes([0.08,0.09,0.18,0.3]),
	               			fig.add_axes([0.32,0.09,0.18,0.3]),
	               			fig.add_axes([0.56,0.09,0.18,0.3]),
	               			fig.add_axes([0.80,0.09,0.18,0.3])])
	    
	for a in top_axes:
	    a.axvline(120, color='black', alpha=0.5)
	    a.axvline(140, color='black', alpha=0.5)
	    a.axvline(160, color='black', alpha=0.5)    

	# define test set (optionally on GPU)
	NSamp = X.shape[0] #by defauls generate as many samples as in training set
	testX = np.linspace(0,1,NSamp)
	test_x = torch.tensor(testX).float().cuda(device=device)

	Y_sim = generate_test_samples(model, test_x)
	    
	utils.Plot_MixModel_Param_MCMC(top_axes,model,test_x,testX*160,rho=plot_pearson(X,Y),title=' for {} vs {}'.format(name_x,name_y))

	bottom_axes[0].set_ylabel(name_y)
	bottom_axes[0].set_xlabel(name_x)

	interval_ends = [0,60,120,140,160]
	utils.Plot_Copula_Density(bottom_axes, testX.squeeze()*160, Y_sim.squeeze(), interval_ends, shade=True)
	utils.Plot_Copula_Density(bottom_axes, X.squeeze()*160, Y, interval_ends, shade=False, color='#073763ff')

	plt.subplots_adjust(wspace=0.5)

	fig.savefig(filename)
	plt.close()

def select_copula_model(X: Tensor, Y: Tensor, device: torch.device,
						exp_pref: str, path_output: str, name_x: str, name_y: str):

	t_start = time.time()

	elements = [bvcopula.GaussianCopula_Likelihood(),
				bvcopula.FrankCopula_Likelihood(),
				bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
				bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
				bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
				bvcopula.ClaytonCopula_Likelihood(rotation='270°'),
				bvcopula.GumbelCopula_Likelihood(rotation='0°'),
				bvcopula.GumbelCopula_Likelihood(rotation='90°'),
				bvcopula.GumbelCopula_Likelihood(rotation='180°'),
				bvcopula.GumbelCopula_Likelihood(rotation='270°')
				]

	#layer0
	waics = np.zeros(len(elements))
	for i, element in enumerate(elements):
		waic = -float("Inf")
		try:
			exp_name = exp_name = '{}_{}-{}'.format(exp_pref,name_x,name_y)
			name = '{}_{}'.format(exp_name,element.name+strrot(element.rotation))
			plot_loss = '{}/loss_{}.png'.format(path_output,name)
			waic, model = infer([element],X,Y,device=device,output_loss=plot_loss)
			plot_res = '{}/res_{}.png'.format(path_output,name)
			plot_fit(model, X, Y, name_x, name_y, plot_res, device=device)
		except ValueError as error:
			print(error)
			print(element.name,' failed')
		finally:
			waics[i] = waic
	# waics = np.ones(len(elements)) * (-float("Inf"))
	# waics[1] = -3.0
	best_ind = np.argmax(waics)
	best = elements[best_ind]
	print("Best single copula: {} {} (WAIC = {:.3})".format(best.name,strrot(best.rotation),np.max(waics)))

	#find best copulas for corners
	best_corners=''
	st = 2
	for i in range(4):
		assert elements[st+i].name=='Clayton'
		assert elements[st+i+4].name=='Gumbel'
		if waics[st+i]>waics[st+i+4]:
			best_corners+='C'
		elif waics[st+i]<waics[st+i+4]:
			best_corners+='G'
		else:
			best_corners+='X'
	print('Best corners: ',best_corners)

	#output = generate_test_functions(model, X.shape[0],DEVICE=DEVICE)
	#important_copulas = important_copulas(model,output)

	# #mask inavailable elements out
	# gauss_frank_count = 0
	# available = np.ones(len(elements)).astype("bool")

	# (available,gauss_frank_count) = update_availability(best_ind,available,gauss_frank_count)

	# best_waics_layers = np.ones(MAX_MIX)* (-float("Inf"))
	# actual_num_copulas = np.zeros(MAX_MIX)
	# best_waics_layers[0] = np.max(waics)
	# actual_num_copulas[0] = 1
	# sequence = [best]

	# best_name = '{}_{}'.format(exp_name,best.name+strrot(best.rotation))
	# new_name = '{}_c{}'.format(exp_name,0)
	# save_best(best_name,new_name,waics,available,sequence)

	t_end = time.time()

	total_time = t_end-t_start
	hours = int(total_time/60/60)
	minutes = int(total_time/60)%60
	seconds = (int(total_time)%(60))
	print("Took {} h {} min {} s ({})".format(hours,minutes,seconds,int(total_time)))