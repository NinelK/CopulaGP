import os
import numpy as np

repeats = 5

for num_particles in [64,128,256,512]:
	for n in np.exp(np.arange(3,5.3,0.333334)*np.log(10)).astype("int"): #6
		max_grid_power = np.clip(0,10,int(np.log(n)/np.log(2))) # no more than 10
		max_grid_size = 2**max_grid_power
		print("***")
		print("NSamp = {}, max_grid: {}".format(n,max_grid_size))
		for m in [2**x for x in range(6,max_grid_power+1)]: #44
			for _ in range(repeats):
				os.system("python GaussCopulaPerformance.py 1 {} {} {}".format(m,n,num_particles))
		#os.system("python GaussCopulaPerformance.py 0 0 {} {}".format(n,num_particles))
