import time
import os
import pickle as pkl
from torch import device
import numpy as np

import utils
import select_copula

animal = 'ST263'
dayN = 1
day_name = 'Day{}'.format(dayN)
path2data = '/home/nina/VRData/Processing/pkls'

exp_pref = '{}_{}'.format(animal,day_name)

out_dir = './out/'+exp_pref
try:
    os.mkdir(out_dir)
except FileExistsError as error:
    print(error)

NN = 23 #number of neurons
beh = 5
results = np.empty((NN+beh,NN+beh),dtype=object)
unique_id = int(np.random.rand()*10000)

for n1 in range(-4,2):
	for n2 in range(n1+1,NN):
		if True:
			X,Y = utils.load_experimental_data(path2data, animal, day_name, n1, n2)

			print('Selecting {}-{}'.format(n1,n2))
			try:
				t_start = time.time()
				(likelihoods, waic) = select_copula.select_copula_model(X,Y,device('cuda:0'),'',out_dir,n1,n2)
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


