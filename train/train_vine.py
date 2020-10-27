from utils import standard_loader
from train import train_next_tree
from train import save_checkpoint, load_checkpoint, save_final

def train_vine(exp, path_data, path_models, path_final,
		layers_max=-1,start=0,gauss=False,device_list=['cpu']):

	X,Y = standard_loader(path_data(0))

	# figure out how many trees to train
	layers = Y.shape[-1]-1 if layers_max == -1 else layers_max

	if start == 0:
		to_save = {}
		to_save['models'], to_save['waics'] = [],[]
	else:
		X,Y,to_save = load_checkpoint(start,path_data(layer),path_models(layer-1))
	for layer in range(start,layers):
		print(f'Starting {exp} layer {layer}/{layers}')
		model, waic, Y = train_next_tree(X,Y,exp,layer,device_list,gauss=gauss)
		to_save['models'].append(model)
		to_save['waics'].append(waic)
		# save checkpoint
		save_checkpoint(X,Y,to_save,path_data(layer+1),path_models(layer))

	save_final(path_data(0),path_models(layers-2),path_final)

	return to_save