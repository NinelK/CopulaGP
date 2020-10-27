import pickle as pkl
from utils import standard_loader, standard_saver

def save_checkpoint(X,Y,to_save,path2data,path2models): 
	standard_saver(path2data,X,Y)
	with open(path2models,"wb") as f:
		pkl.dump(to_save,f)

def load_checkpoint(layer,path2data,path2models): 
	X,Y = standard_loader(path2data)
	with open(path2models,"rb") as f:
		d = pkl.load(f)
	return X,Y,d

def save_final(path2data,path2models,path_final): 
	'''
	Adds final trained model
	(trained up to a specified layer (=tree))
	to the training dataset
	'''
	with open(path2data,"rb") as f:
		d0 = pkl.load(f)
	with open(path2models,"rb") as f:
		df = pkl.load(f)
	
	with open(path_final,"wb") as f:
		pkl.dump(d0.update(df),f)