import pickle as pkl
import torch
import numpy as np
import sys
from . import conf
sys.path.insert(0, conf.path2code)

import utils

def transform2next_layer(exp_pref,layer,device):

	path_models = f"{conf.path2outputs}/{exp_pref}"

	print(f"Tranforming {exp_pref} layer {layer} -> {layer+1}...")

	# Assume we were have a good, uncorrupted list of models in a single file
	with open(f"{path_models}/models_layer{layer}.pkl","rb") as f:
	    model_list=pkl.load(f)
	layer_width = model_list.shape[0]
	# Now load the inputs
	X,Y = utils.standard_loader(f"{conf.path2data}/{exp_pref}/{exp_pref}_layer{layer}.pkl")
	NN = Y.shape[-1]-1
	assert layer_width==NN

	# transform all
	def transform(X,Y,model_list):
	    S = torch.tensor(X).float().squeeze().to(device)
	    next_layer = []
	    for n in range(1,Y.shape[-1]):
	        # load model
	        likelihoods = model_list[n-1]
	        if (len(likelihoods[0])==1) and (likelihoods[1]=='Independence'):
	            next_layer.append(Y[:,n])
	        else:
	            weights_file = f"{path_models}/layer{layer}/model_{exp_pref}_{layer}-{n+layer}.pth"
	            model = utils.get_model(weights_file, likelihoods[0], device) 
	            # load data
	            samples = torch.tensor(Y[:,[n,0]]).float().squeeze().to(device) # order!

	            with torch.no_grad():
	                f_samples = model.gp_model(S).mean#rsample(torch.Size([10]))

	            copula = model.likelihood.get_copula(f_samples)

	            next_layer.append(copula.ccdf(samples).cpu().numpy())
	    return X,next_layer
	X,next_layer = transform(X,Y,model_list)

	# next_layer.append(Y[:,-1])
	assert len(next_layer)==NN

	y = np.array(next_layer).T
	assert np.all(y>=0) & np.all(y<=1)

	data = {}
	data['X'] = X
	data['Y'] = y

	with open(f"{conf.path2data}/{exp_pref}/{exp_pref}_layer{layer+1}.pkl","wb") as f:
	    pkl.dump(data,f)

	print('Transform finished and saved.')

	return NN
