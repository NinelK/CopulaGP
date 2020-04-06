import glob
import os
import numpy as np
import pickle as pkl
from . import conf

def merge_results(exp_pref,layer):
	list_files = glob.glob(f"{conf.path2outputs}/{exp_pref}/layer{layer}*.pkl")

	with open(list_files[0],"rb") as f:
		res = pkl.load(f)
	if len(list_files)>1:
		for file in list_files[1:]:
			with open(file,"rb") as f:
				res_add = pkl.load(f)
			assert res.shape==res_add.shape
			assert np.sum((res!=None) & (res_add!=None)) == 0

			res[res_add!=None] = res_add[res_add!=None]	 
	
	if np.any(res==None):
		print('Missing models: ',layer,' vs. ',np.arange(layer+1,layer+len(res)+1)[res==None])
	else:
		print(f'Layer{layer} is complete and models from different devices merged')
		with open(f"{conf.path2outputs}/{exp_pref}/models_layer{layer}.pkl","wb") as f:
			pkl.dump(res,f)

		# for file in list_files:
		# 	os.remove(file)