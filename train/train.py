import sys
import argparse
import conf
sys.path.insert(0, conf.path2code)
from utils import standard_loader, standard_saver
import train
import pickle as pkl

parser = argparse.ArgumentParser(description='Train copula vine model')
parser.add_argument('-exp', default='', help='Experiment name')
parser.add_argument('-layers', default=-1, help='How many layers? (-1 = all possible)', type=int)
parser.add_argument('-start', default=0, help='Start from a certain layer', type=int)
parser.add_argument('--gauss','-g', default=False, help='Train with only Gauss Copulas', action='store_true')
# TODO paths to exps

args = parser.parse_args()

g = 'G' if args.gauss else ''
path_data = lambda layer: f"{conf.path2data}/outputs/{args.exp}{g}_layer{layer}.pkl"
path_models = lambda layer: f"{conf.path2data}/outputs/{args.exp}{g}_models_layer{layer}.pkl"
path_final = f"{conf.path2data}/outputs/{args.exp}{g}_trained.pkl"

gpu_list = range(2)

X,Y = standard_loader(path_data(0))

# figure out how many trees to train
layers = Y.shape[-1]-1 if args.layers == -1 else args.layers

def save_checkpoint(X,Y,to_save,layer):
	standard_saver(path_data(layer+1),X,Y)
	with open(path_models(layer),"wb") as f:
		pkl.dump(to_save,f)

def load_checkpoint(layer):
	X,Y = standard_loader(path_data(layer))
	with open(path_models(layer-1),"rb") as f:
		d = pkl.load(f)
	return X,Y,d

def save_final():
	'''
	Adds final trained model
	(trained up to a specified layer (=tree))
	to the training dataset
	'''
	with open(path_data(0),"rb") as f:
		d0 = pkl.load(f)
	with open(path_models(layers-2),"rb") as f:
		df = pkl.load(f)
	
	with open(path_final,"wb") as f:
		pkl.dump(d0.update(df),f)


if args.start == 0:
	to_save = {}
	to_save['models'], to_save['waics'] = [],[]
else:
	X,Y,to_save = load_checkpoint(args.start)
for layer in range(args.start,layers):
	print(f'Starting {args.exp} layer {layer}/{layers}')
	model, waic, Y = train.train_next_tree(X,Y,args.exp,layer,gpu_list,gauss=args.gauss)
	to_save['models'].append(model)
	to_save['waics'].append(waic)
	# save checkpoint
	save_checkpoint(X,Y,to_save,layer)

save_final()
