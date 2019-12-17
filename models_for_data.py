import time
import os
import pickle as pkl
from torch import device
import numpy as np
import sys
import multiprocessing

import utils
import select_copula

gpu_id_list = range(1) 
#[i//2 for i in range(8*2)]  # 2 workers on each GPU

animal = 'ST264'
dayN = 1
day_name = 'Day{}'.format(dayN)
path2data = '/home/nina/VRData/Processing/pkls'

exp_pref = '{}_{}'.format(animal,day_name)

out_dir = 'out_london/'+exp_pref
try:
	os.mkdir(out_dir)
except FileExistsError as error:
	print(error)

NN = 34 #number of neurons
beh = 5

def worker(n1,n2):
	#get unique gpu id for cpu id
	cpu_name = multiprocessing.current_process().name
	cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
	gpu_id = gpu_id_list[cpu_id]

	device_str = 'cuda:{}'.format(gpu_id)

	unique_id = cpu_id
	results = np.empty((NN+beh,NN+beh),dtype=object)

	X,Y = utils.load_experimental_data(path2data, animal, day_name, n1, n2)

	print('Selecting {}-{} on {}'.format(n1,n2,device_str))
	try:
		t_start = time.time()
		(likelihoods, waic) = select_copula.select_copula_model(X,Y,device(device_str),'',out_dir,n1,n2)
		t_end = time.time()
		print('Selection took {} min'.format(int((t_end-t_start)/60)))
	except RuntimeError as error:
		print(error)
	finally:
		path2model = "{}/{}-{}.pkl".format(out_dir,n1,n2)   
		with open(path2model,'wb') as f:
			pkl.dump(likelihoods,f)

		with open(out_dir+'_model_list.txt','a') as f:
			f.write("{}-{} {}\t{:.0f}\t{}\n".format(n1,n2,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)))

		results[beh+n1,beh+n2] = [likelihoods,utils.get_copula_name_string(likelihoods),waic,int(t_end-t_start)]

		with open('{}_{}_models.pkl'.format(out_dir,unique_id),'wb') as f:
			pkl.dump(results,f)   

	return 0

if __name__ == '__main__':
	pool = multiprocessing.Pool(len(gpu_id_list))

	for n1 in range(0,NN-1):
		for n2 in range(n1+1,NN):
			res = pool.apply_async(worker, (n1,n2,))
	pool.close()
	pool.join()  # block at this line until all processes are done
	print("completed")
		


