import sys
import shutil
import numpy as np
import pickle as pkl
import matplotlib.cm as cm
import time
import torch
import gpytorch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

#from bvcopula import SingleParamCopulaBase
from bvcopula import GaussianCopula, GaussianCopula_Likelihood, GaussianCopula_Flow_Likelihood
from bvcopula import FrankCopula, FrankCopula_Likelihood
from bvcopula import ClaytonCopula, ClaytonCopula_Likelihood
from bvcopula import GumbelCopula, GumbelCopula_Likelihood
from bvcopula import StudentTCopula, StudentTCopula_Likelihood
from bvcopula import MixtureCopula, MixtureCopula_Likelihood
from bvcopula import GPInferenceModel, KISS_GPInferenceModel
from bvcopula import Mixed_GPInferenceModel, GridInterpolationVariationalStrategy

path = '/home/nina/VRData/Processing/pkls'
path_output = './data_scan/'
base_lr = 1e-3
iter_print = 100
NUM_ITER = 2000

def load_data(animal,day_name,n1,n2):

	def data_from_n(n):
		if n>0:
			data = signals[n]
		elif n==-1:
			data = behaviour_pkl['transformed_velocity']
		elif n==-2:
			data = behaviour_pkl['transformed_licks']
		elif n==-3:
			data = behaviour_pkl['transformed_early_reward'] + behaviour_pkl['transformed_late_reward']
		elif n==-4:
			data = behaviour_pkl['transformed_early_reward']
		elif n==-5:
			data = behaviour_pkl['transformed_late_reward']
		else:
			raise ValueError('n is out of range')
		return data

	with open("{}/{}_{}_signals.pkl".format(path,animal,day_name),'rb') as f:
	    signal_pkl = pkl.load(f)
	with open("{}/{}_{}_behaviour.pkl".format(path,animal,day_name),'rb') as f:
	    behaviour_pkl = pkl.load(f)
	for s in ['ROIsN','trialStart','maxTrialNum','trials']:
	    assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))

	signals = signal_pkl['signals_transformed']

	data1 = data_from_n(n1)
	data2 = data_from_n(n2)

	Y_all = np.array([data1,data2]).T
	X_all = np.array(behaviour_pkl['position'])#local_time

	rule = (Y_all[:,0]>1e-4) & (Y_all[:,1]>1e-4)  \
	        & (Y_all[:,0]<1.-1e-4) & (Y_all[:,1]<1.-1e-4)
	 
	X = np.reshape(X_all[rule],(-1,1))
	X[X<0] = 160.+X[X<0]
	X[X>160] = X[X>160]-160.
	X = X/160.
	Y = Y_all[rule]
	
	return X, Y

def plot_loss(filename, losses, rbf, means):
	# prot loss function and kernel length
	fig, (loss, kern, mean) = plt.subplots(1,3,figsize=(15,2))
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

def plot_result_param(filename, model, likelihoods, test_x, testX, NSamp):
	# visualize the result
	fig, ax = plt.subplots(1,2,figsize=(12, 4))
	colors = cm.rainbow(np.linspace(0., 1., NSamp))

	with torch.no_grad():
	    output = model(test_x)
	    
	gplink = model.likelihood.gplink_function

	thetas, mixes = gplink(output.mean)
	lower, upper = output.confidence_region() #thetas & mix together
	thetas_low, mixes_low = gplink(lower)
	thetas_upp, mixes_upp = gplink(upper)

	copulas = [lik.name for lik in likelihoods]
	rotations = [lik.rotation for lik in likelihoods]
	 
	for t,l,u,c,r in zip(thetas,thetas_low,thetas_upp,copulas,rotations):
	    F_mean = t.detach().cpu().numpy()
	    line, = ax[0].plot(testX, F_mean, label = '{} {}'.format(c,r))
	    ax[0].fill_between(testX, l.detach().cpu().numpy(),
	                    u.detach().cpu().numpy(), color=line.get_color(), alpha=0.5)
	    
	ax[0].scatter(np.linspace(0., 1., NSamp),np.zeros(NSamp),color=colors)
	ax[0].set_xlabel('x')
	ax[0].set_ylabel('gp_link(f)')
	ax[0].set_title('Copula parameters (thetas)')
	ax[0].legend()

	_, sampled = gplink(output.rsample(torch.Size([100])))

	for i, (t,c,r) in enumerate(zip(mixes[:],copulas,rotations)):
	    F_mean = t.detach().cpu().numpy()
	    line, = ax[1].plot(testX, F_mean, label = '{} {}'.format(c,r))
	    sm = np.mean(sampled[i].cpu().numpy(),axis=0)
	    std = np.std(sampled[i].cpu().numpy(),axis=0)
	    ax[1].fill_between(testX, sm-std, sm+std, color=line.get_color(), alpha=0.5)

	ax[1].scatter(np.linspace(0., 1., NSamp),np.zeros(NSamp),color=colors)
	ax[1].set_xlabel('x')
	ax[1].set_ylabel('gp_link(f)')
	ax[1].set_title('Mixing concentrations')
	ax[1].legend()

	fig.savefig(filename)
	plt.close()

def plot_result_copula(filename, model, likelihoods, X, Y, train_x):
	with gpytorch.settings.num_likelihood_samples(1):
	    gplink = model.likelihood.gplink_function
	    copulas = [lik.copula for lik in likelihoods]
	    rotations = [lik.rotation for lik in likelihoods]
	    thetas, mixes = gplink(model(train_x).mean)
	    Y_sim = model.likelihood.copula(thetas,mixes,copulas,rotations=rotations).rsample().cpu().detach().numpy()

	fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(10,5))
	fig.subplots_adjust(hspace=0.5)

	X_pos = X.squeeze()*160
	Y_pos = Y_sim.squeeze()

	mrg = 0.2

	for s,e,ax,name in zip([0,60,120,140],[60,120,140,160],axes.flatten(),['0-60','60-120','120-140','140-160']): #['0-1','1-2','2-3','3-4']
	    sns.kdeplot(*Y_pos[(X_pos[:]>=s) & (X_pos[:]<e)].T, ax=ax, shade=False,  shade_lowest=True)
	    ax.set(title=name, xlim=(-mrg,1+mrg), ylim=(-mrg,1+mrg))
	    
	X_pos = X.squeeze()*160
	Y_pos = Y

	mrg = 0.2

	for s,e,ax,name in zip([0,60,120,140],[60,120,140,160],axes.flatten()[4:],['0-60','60-120','120-140','140-160']): #['0-1','1-2','2-3','3-4']
	    sns.kdeplot(*Y_pos[(X_pos[:]>=s) & (X_pos[:]<e)].T, ax=ax, shade=False,  shade_lowest=True)
	    ax.set(title=name, xlim=(-mrg,1+mrg), ylim=(-mrg,1+mrg))

	fig.savefig(filename)
	plt.close()

def waic(model,likelihoods,train_x,train_y):
	torch.cuda.empty_cache() 

	gplink = model.likelihood.gplink_function
	copulas = [lik.copula for lik in likelihoods]
	rotations = [lik.rotation for lik in likelihoods]

	S=500
	with torch.no_grad():
	    f_samples = model(train_x).rsample(torch.Size([S]))
	    thetas, mixes = gplink(f_samples)

	    log_prob = model.likelihood.copula(thetas,mixes,copulas,rotations=rotations).\
	            log_prob(train_y).detach().cpu().numpy()
	    pwaic = np.var(log_prob,axis=0).sum()
	    sum_prob = np.exp(log_prob).sum(axis=0)
	    lpd=np.sum(np.log(sum_prob)-np.log(S)) # sum_M log(1/N * sum^i_S p(y|theta_i)), where N is train_x.shape[0]

	    #print('Lpd: ',lpd)
	    #print('p_WAIC: ',pwaic)
	    print('WAIC: ', lpd - pwaic)

	del(f_samples, log_prob)
	torch.cuda.empty_cache() 

	return (lpd - pwaic)

def strrot(rotation):
	if rotation is not None:
		return rotation
	else:
		return ''

def infer(likelihoods,X,Y,name, theta_sharing=None):

	if theta_sharing is not None:
		theta_sharing = theta_sharing
		num_fs = len(likelihoods)+thetas_sharing.max().numpy() # indep_thetas + num_copulas - 1
	else:
   		theta_sharing = torch.arange(0,len(likelihoods)).long()
   		num_fs = 2*len(likelihoods)-1

	NSamp = X.shape[0]

	print('Trying ',*[lik.name+' '+strrot(lik.rotation) for lik in likelihoods])

	#convert numpy data to tensors (optionally on GPU)
	train_x = torch.tensor(X).float().cuda(device=0)
	train_y = torch.tensor(Y).float().cuda(device=0)

	# define the model (optionally on GPU)
	grid_size = 128
	if len(likelihoods)>0:
		model = Mixed_GPInferenceModel(
				MixtureCopula_Likelihood(likelihoods, 
									theta_sharing=theta_sharing), 
	                            num_fs,  
	                            prior_rbf_length=0.2, 
	                            grid_size=grid_size).cuda(device=0)
	else:
		model = KISS_GPInferenceModel(likelihoods[0], 
	                               prior_rbf_length=0.2, 
	                               grid_size=grid_size).cuda(device=0)

	# We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
	optimizer = torch.optim.Adam([
	    {'params': model.mean_module.parameters()},
	    {'params': model.variational_strategy.variational_distribution.chol_variational_covar},
	    {'params': model.covar_module.raw_outputscale},
	    {'params': model.variational_strategy.variational_distribution.variational_mean, 'lr': .1},
	    {'params': model.covar_module.base_kernel.raw_lengthscale, 'lr': .01} #, 'lr': 0.001
	], lr=base_lr)

	# train the model

	mll  = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=train_y.size(0), combine_terms=True)

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
	            #loss_scale = np.abs(loss.detach().cpu().numpy() - np.mean(losses))
	            
	            # print('Iter {}/{} - Loss: {:.3}   lengthscale: {}, dLoss: {:.3}, mean f: {:.3}, dmean: {:.3}'.format(
	            #     i + 1, num_iter, loss,
	            #     model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze(), 
	            #     mean_p, np.mean(means[-1]), 
	            #     np.mean(np.abs(means[-100]-means[-1]))
	            # ))

	            if (0 < mean_p < 0.001):# & (np.mean(np.abs(1-means[-100]/means[-1])) < 0.05): 
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

	# define test set (optionally on GPU)
	denser = 1 # make test set 2 times denser then the training set
	testX = np.linspace(0,1,denser*NSamp)
	test_x = torch.tensor(testX).float().cuda(device=0)

	plot_loss('{}loss/loss_{}.png'.format(path_output,name), 
		losses, rbf, means)
	plot_result_param('{}result_param/rp_{}.png'.format(path_output,name), 
		model, likelihoods, test_x, testX, NSamp)
	plot_result_copula('{}result_copula/rc_{}.png'.format(path_output,name), 
		model, likelihoods, X, Y, train_x)

	return waic(model,likelihoods,train_x,train_y)

def save_best(best_name,new_name,waics,available,sequence):
	shutil.copy('{}loss/loss_{}.png'.format(path_output,best_name),
		'{}best/loss_{}.png'.format(path_output,new_name))
	shutil.copy('{}result_param/rp_{}.png'.format(path_output,best_name),
		'{}best/rp_{}.png'.format(path_output,new_name))
	shutil.copy('{}result_copula/rc_{}.png'.format(path_output,best_name),
		'{}best/rc_{}.png'.format(path_output,new_name))
	to_save = {}
	to_save['waics'] = waics
	to_save['available'] = available
	to_save['sequence'] = sequence
	to_save['copulas'] = [[lik.name, lik.rotation] for lik in sequence]
	with open('{}best/{}.pkl'.format(path_output,new_name),'wb') as f:
		pkl.dump(to_save,f)

def update_availability(best_ind,available,gauss_count,frank_count):
	if best_ind>=2:
		if best_ind<=5:
			available[best_ind] = 0
			available[best_ind+4] = 0
		else:
			available[best_ind] = 0
			available[best_ind-4] = 0
	elif best_ind==0:
		gauss_count += 1
	elif best_ind==1:
		frank_count += 1
	return (available,gauss_count,frank_count)

def main():
	animal_number = int(sys.argv[1])
	day_number = int(sys.argv[2])
	n1 = int(sys.argv[3])
	n2 = int(sys.argv[4])

	animal = 'ST{:d}'.format(animal_number)
	day_name = 'Day{:d}'.format(day_number)

	print(animal,' ',day_name)

	exp_name = '{}_{}_{}-{}'.format(animal,day_name,n1,n2)

	X, Y = load_data(animal,day_name,n1,n2)

	t_start = time.time()

	elements = [GaussianCopula_Likelihood(),
				FrankCopula_Likelihood(),
				ClaytonCopula_Likelihood(rotation='0°'),
				ClaytonCopula_Likelihood(rotation='90°'),
				ClaytonCopula_Likelihood(rotation='180°'),
				ClaytonCopula_Likelihood(rotation='270°'),
				GumbelCopula_Likelihood(rotation='0°'),
				GumbelCopula_Likelihood(rotation='90°'),
				GumbelCopula_Likelihood(rotation='180°'),
				GumbelCopula_Likelihood(rotation='270°')]

	#layer0
	waics = np.zeros(len(elements))
	for i, element in enumerate(elements):
		name = '{}_{}'.format(exp_name,element.name+strrot(element.rotation))
		waic = -float("Inf")
		try:
			waic = infer([element],X,Y,name)
		except ValueError as error:
			print(error)
			print(element.name,' failed')
		finally:
			waics[i] = waic
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

	#mask inavailable elements out
	gauss_count = 0
	frank_count = 0
	available = np.ones(len(elements)).astype("bool")

	(available,gauss_count,frank_count) = update_availability(best_ind,available,gauss_count,frank_count)

	simpler_waic = np.max(waics)
	sequence = [best]

	best_name = '{}_{}'.format(exp_name,best.name+strrot(best.rotation))
	new_name = '{}_c{}'.format(exp_name,0)
	save_best(best_name,new_name,waics,available,sequence)

	for layer in range(5): #6 copulas is more than enough
		waics = np.ones(len(elements)) * (-float("Inf"))
		for i in np.arange(len(elements))[available]:
			element = elements[i]
			copulas_names=''
			for lik in (sequence+[element]):
				copulas_names += lik.name+strrot(lik.rotation)
			name = '{}_{}'.format(exp_name,copulas_names)
			waic = -float("Inf")
			try:
				likelihoods = sequence+[element]
				waic = infer(likelihoods,X,Y,name)
			except ValueError as error:
				print(error)
				print(element.name,' failed')
			finally:
				waics[i] = waic

		if np.max(waics)>simpler_waic:
			best_ind = np.argmax(waics)
			best = elements[best_ind]
			print("Best added copula: {} {} (WAIC = {:.3})".format(best.name,strrot(best.rotation),np.max(waics)))

			(available,gauss_count,frank_count) = update_availability(best_ind,available,gauss_count,frank_count)

			simpler_waic = np.max(waics)
			sequence = sequence + [best]

			copulas_names=''
			for lik in sequence:
				copulas_names += lik.name+strrot(lik.rotation)
			best_name = '{}_{}'.format(exp_name,copulas_names)
			new_name = '{}_c{}'.format(exp_name,layer+1)
			save_best(best_name,new_name,waics,available,sequence)

			elements_available=[]
			for i in range(len(available)):
				if available[i]:
					elements_available.append(elements[i])
			print('Availability: ',*[lik.name+strrot(lik.rotation) for lik in elements_available])
			del(elements_available)
			if np.all(available==0):
				print("Search is done, nothing to add.")	
				break
		else:
			print("Search is done, best copula found.")
			break

	t_end = time.time()
			
	print("Best final sequence is: ",*[lik.name+strrot(lik.rotation) for lik in sequence])
	total_time = t_end-t_start
	hours = int(total_time/60/60)
	minutes = int(total_time/60)%60
	seconds = (int(total_time)%(60))
	print("Took {} h {} min {} s ({})".format(hours,minutes,seconds,int(total_time)))
	
if __name__ == "__main__":
	main()