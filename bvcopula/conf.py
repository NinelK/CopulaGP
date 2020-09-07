#learning rates
grid_size = 60 # the size used for model selection
# fine_grid_size = 120 # the size used for final model
base_lr = 0.05
hyper_lr = 0.02
iter_print = 100
max_num_iter = 500
loss_tol = 0.005 #the minimal change in loss that indicates convergence
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
Gumbel_Theta_Max = 10.0		# (1, max)
# looser limits for sample generation
Frank_Theta_Sampling_Max = 88.0 	# (-max, max)
Clayton_Theta_Sampling_Max = 22.5 	# (0, max)
Gumbel_Theta_Sampling_Max = 16.0 	# (1, max) #no clear critical value here, it is around 16

#Gaussian full dependence
Gauss_diag = 1e-5 # how far from diagonal the point can be to be considered as u==v

# # how we found max_theta_sampling
# a = []
# thetas = np.arange(21.0,25.0,.2)
# for theta in thetas:
#     copula = bvcopula.ClaytonCopula(torch.tensor([theta]))
#     bin_size = 50
#     #generate samples
#     S = copula.sample(torch.Size([100000])).numpy().squeeze()
#     S = S.reshape(-1,2)
#     r_den = np.histogram2d(*S.T,bins=[bin_size,bin_size],density=True)[0]
#     a.append(np.max(r_den - r_den.T))
# plt.plot(thetas,a)


# waic parameters
waic_samples = 500
waic_resamples = 3 #how many times to repeat
