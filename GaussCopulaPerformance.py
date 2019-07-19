import math
import torch
import pyro
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from bvcopula import GaussianCopula, GaussianCopula_Likelihood, GaussianCopula_Flow_Likelihood, GPInferenceModel, KISS_GPInferenceModel
import matplotlib.cm as cm
from pyro import optim
import time
import sys

CUDA = 1
DEVICE = 1
FLOW = 0
NSamp = 100
KISS = 1
grid_size = 128
num_particles = 64
early_stopping_threshold = 0.02
denser = 2 # make test set 2 times denser then the training set

iter_print = 100

if KISS == 0:
	base_lr = .1/NSamp
	kernel_lr = 1./NSamp
	mean_lr = 1./NSamp
else:
	base_lr = 1e-3
	kernel_lr = .05*(grid_size/128)
	mean_lr = .1

if FLOW ==0:
	likelihood = GaussianCopula_Likelihood()
	print('Noflow')
else:
	likelihood = GaussianCopula_Flow_Likelihood()
	print('Flow')

#here we specify a 'true' latent function lambda
scale = lambda x: np.sin(2*np.pi*1*x)*1.+0.0

#setup color scheme for samples
colors = cm.rainbow(np.linspace(0, 1, NSamp))

def generate_data():

	# here we generate some synthetic samples
	X = np.linspace(0,1,NSamp) 
	scales = [scale(x) for x in X]
	model = GaussianCopula(torch.tensor(scales).float())
	Y = model.sample().numpy().squeeze()

	# define test set (optionally on GPU)
	testX = np.linspace(0,1,denser*NSamp)

	#convert numpy data to tensors (optionally on GPU)
	if CUDA == 0:
		train_x = torch.tensor(X).float()
		train_y = torch.tensor(Y).float()
		test_x = torch.tensor(testX).float()
		# define the model (optionally on GPU)
		if KISS == 0:
			model = GPInferenceModel(train_x, train_y, likelihood)
		elif KISS == 1:
			model = KISS_GPInferenceModel(likelihood, grid_size=grid_size)
		else:
			print("Specify correct cuda flag.")

	elif(CUDA == 1):
		train_x = torch.tensor(X).float().cuda(device=DEVICE)
		train_y = torch.tensor(Y).float().cuda(device=DEVICE)
		test_x = torch.tensor(testX).float().cuda(device=DEVICE)
		# define the model (optionally on GPU)
		if KISS == 0:
			model = GPInferenceModel(train_x, train_y, likelihood).cuda(device=DEVICE)
		elif KISS == 1:
			model = KISS_GPInferenceModel(likelihood, grid_size=grid_size).cuda(device=DEVICE)
		else:
			print("Specify correct cuda flag.")

	else:
		print("Specify correct cuda flag.")

	return (train_x, train_y, model, test_x, X, Y, testX)

def setup_training(train_x, train_y, model):
	# train the model

	# set learning rates for different hyperparameters
	def per_param_callable(module_name, param_name):
	    if param_name == 'covar_module.base_kernel.raw_lengthscale':
	        return {"lr": kernel_lr} #.1 for 256 particles
	    elif param_name == 'variational_strategy.variational_distribution.variational_mean':
	        return {"lr": mean_lr}
	    else:
	        return {"lr": base_lr}

	# Use the adam optimizer
	optimizer = optim.Adam(per_param_callable)

	pyro.clear_param_store() # clean run

	losses, rbf = [], []

	def train(num_iter=5000):
	    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, max_plate_nesting=1)
	    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)
	    model.train()

	    p = 0.
	    final_step = num_iter
	    for i in range(num_iter):
	        model.zero_grad()
	        loss = svi.step(train_x, train_y)
	        if len(losses)>100: 
	            p += np.abs(np.mean(losses[-50:]) - np.mean(losses[-100:-50]))
	        losses.append(loss)
	        rbf.append(model.covar_module.base_kernel.lengthscale.item())
	        if not (i + 1) % iter_print:
	            mean_p = p/100
	            # print('Iter {}/{} - Loss: {:.3}   lengthscale: {:.3}, Rel_dLoss: {:.3}'.format(
	            #     i + 1, num_iter, loss,
	            #     model.covar_module.base_kernel.lengthscale.item(), mean_p/np.abs(loss)
	            # ))
	            #print(np.std(losses[-100:]))
	            if 0 < mean_p/np.abs(loss) < early_stopping_threshold: 
	                print("Converged!")
	                final_step = i + 1
	                break
	            p = 0.
	    return final_step
	ts = time.time()
	final_step = train()
	te = time.time()

	return (losses, rbf, ts, te, final_step)


def plot_loss(filename, losses, rbf):
	# prot loss function and kernel length
	fig, (loss, kern) = plt.subplots(1,2,figsize=(12,2))
	loss.plot(losses)
	loss.set_xlabel("Epoch #")
	loss.set_ylabel("Loss")
	loss.set_ylim(np.min(losses)*1.1,20)
	kern.plot(rbf)
	kern.set_xlabel("Epoch #")
	kern.set_ylabel("Kernel scale parameter")
	fig.savefig(filename)

def plot_results(filename, test_x, testX, X, Y):
	# visualize the result
	fig, (func, sim, true) = plt.subplots(1,3,figsize=(15, 3))

	model.eval()
	with torch.no_grad():
	    output = model(test_x)
	    
	gplink = model.likelihood.gplink_function
	    
	F_mean = gplink(output.mean).detach().cpu().numpy()
	line, = func.plot(testX, F_mean, label = 'GP prediction')
	lower, upper = output.confidence_region()
	func.fill_between(testX, gplink(lower).detach().cpu().numpy(),
	                gplink(upper).detach().cpu().numpy(), color=line.get_color(), alpha=0.5)

	func.plot(testX,scale(testX), label = 'True latent function')
	func.scatter(X,np.zeros_like(X),color=colors)
	func.set_xlabel('x')
	func.set_ylabel('gp_link(f)')
	func.set_title('Latent function')
	func.legend()

	# sample from p(y|D,x) = \int p(y|f) p(f|D,x) (doubly stochastic)
	with gpytorch.settings.num_likelihood_samples(1):
	    Y_sim = model.likelihood(model(train_x)).rsample().cpu().detach().numpy()
	true.scatter(*Y[::skip].T, label = 'True train data', color=colors[::skip])
	sim.scatter(*Y_sim[:,:,::skip].T, label = 'Sample from the model', color=colors[::skip])
	for ax in [sim,true]:
	    ax.set_xlabel('$y_1$')
	    ax.set_ylabel('$y_2$')
	sim.set_title('Samples from copula with theta=gplink(f(x))')
	true.set_title('True data samples')
	# samp.legend()

	fig.savefig(filename)

def find_logprob(model, train_x, train_y):
	gplink = model.likelihood.gplink_function
	thetas = gplink(model(train_x).rsample(sample_shape=torch.Size([1000])))
	log_probs = torch.sum(GaussianCopula(thetas).log_prob(train_y),dim=1)
	if CUDA == 1:
		log_probs = log_probs.cpu()
	log_probs = log_probs.detach().numpy()

	mean_ll = np.mean(log_probs)/NSamp

	ll_of_mean_f = torch.sum(GaussianCopula(gplink(model(train_x).mean)).log_prob(train_y).detach())

	if CUDA == 1:
		ll_of_mean_f = ll_of_mean_f.cpu()
	ll_of_mean_f = ll_of_mean_f.numpy()/NSamp

	return mean_ll, ll_of_mean_f


def yesno(x):
	if x==0:
		return "NO"
	elif x==1:
		return "YES"
	else:
		return "What?!"

if __name__ == '__main__':

	KISS = int(sys.argv[1])
	grid_size = int(sys.argv[2])
	NSamp = int(sys.argv[3])
	num_particles = int(sys.argv[4])

	if NSamp > 1000:			#outputting more than 1000 points does not look great
		skip = int(NSamp/1000)
	else:
		skip = 1

	#change colorscheme for new NSamp
	colors = cm.rainbow(np.linspace(0, 1, NSamp))

	def detailes_print(cuda, device, word):
		if cuda==0:
			return "NO"
		elif cuda==1:
			return "YES ({}: {})".format(word,device)
		else:
			return "What?!"

	print("CUDA: {}; KISS: {};".format(detailes_print(CUDA, DEVICE, "DEVICE"),detailes_print(KISS, grid_size, "grid_size")))
	print("LR: {}; THR: {}; Particles: {};".format(base_lr,early_stopping_threshold,num_particles))
	print("N: {}".format(NSamp))

	t1 = time.time()
	train_x, train_y, model, test_x, X, Y, testX = generate_data()
	t2 = time.time()
	losses, rbf, t3, t4, steps = setup_training(train_x, train_y, model)

	if KISS == 0:
		code = "Exact_{}_{}".format(NSamp,num_particles)
	else:
		code = "Kiss_{}_{}_{}".format(grid_size,NSamp,num_particles)

	plot_loss("./logs/loss_{}.png".format(code), losses, rbf)
	plot_results("./logs/results_{}.png".format(code), test_x, testX, X, Y)
	t5 = time.time()

	# print("Generating data took {} ms".format(int(1e3*(t2-t1))))
	# print("Testing and output took {} ms".format(int(1e3*(t5-t4))))
	#print("Setting up optimizer took {}".format(t3-t2)) #nothing
	print("Training took {} min {} s (={} s)".format(int((t4-t3)/60),int((t4-t3)%60),int(t4-t3)))
	print("Steps: {}; Time/step: {} ms/step".format(steps,int(1e3*(t4-t3)/steps)))

	mean_ll, ll_of_mean_f = find_logprob(model, train_x, train_y)

	print("Mean doubly-stoch LL: {:.3}; LL of E[f]: {:.3};".format(mean_ll, ll_of_mean_f))

	grid_size_ = grid_size
	if KISS == 0:
		grid_size_ = 0

	with open("./logs/log_cuda.csv","a") as f:
		f.write("{},{},{},{},{},{},{},{},{}\n".format(KISS,NSamp,grid_size_,num_particles,int(t4-t3),steps,mean_ll,ll_of_mean_f,losses[-1]))
