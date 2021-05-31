#learning rates
grid_size = 60 # the size used for model selection
# fine_grid_size = 120 # the size used for final model
base_lr = 0.05 
hyper_lr = 0.02 
decrease_lr = 1. 
mix_lr_ratio = .5 # lr_mix / lr_thetas
iter_print = 100
max_num_iter = 1500 
a_loss_tol = 0.0001 		# the minimal change in loss that indicates convergence
r_loss_tol = 0.001 			# or, the relative change in loss
loss_tol2check_waic = 0.05
waic_tol = 0.005 # maximal WAIC indistinguishable from 0
loss_av = 25 # average over this number x 2 of epochs is used for early stopping

# copula's theta ranges
# here thetas are mainly constrained by the summation of probabilities in mixture model,
# which should not become +inf
Gauss_Safe_Theta = 0.9999	# (-safe,+safe), for safe mode, otherwise (-1,1)
Frank_Theta_Max = 13.0 		# (-max, max) Reduced due to bad ccdf
Frank_Theta_Flip = 9.0
Clayton_Theta_Max = 9.4 	# (0, max)
Gumbel_Theta_Max = 9.0		# (1, max)
# looser limits for sample generation
Frank_Theta_Sampling_Max = 88.0 	# (-max, max)
Clayton_Theta_Sampling_Max = 22.5 	# (0, max)
Gumbel_Theta_Sampling_Max = 16.0 	# (1, max) #no clear critical value here, it is around 16

#Gaussian full dependence
Gauss_diag = 1e-5 # how far from diagonal the point can be to be considered as u==v

# waic parameters
waic_samples = 500
waic_resamples = 3 #how many times to repeat
