import os
import numpy as np

for n in np.exp(np.arange(1,5,0.333334)*np.log(10)).astype("int"):
	max_grid_power = int(np.log(n)/np.log(2))
	max_grid_size = 2**max_grid_power
	print("***")
	print("NSamp = {}, max_grid: {}".format(n,max_grid_size))
	for m in [2**x for x in range(3,max_grid_power+1)]:
		os.system("python GaussCopulaPerformance.py 1 {} {}".format(m,n))
	os.system("python GaussCopulaPerformance.py 0 0 {}".format(n))