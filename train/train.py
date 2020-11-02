import sys
import argparse
import conf
import time
sys.path.insert(0, conf.path2code)
from train import train_vine

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train copula vine model')
	parser.add_argument('-exp', default='', help='Experiment name')
	parser.add_argument('-layers', default=-1, help='How many layers? (-1 = all possible)', type=int)
	parser.add_argument('-start', default=0, help='Re-Start from a certain layer', type=int)
	parser.add_argument('--gauss','-g', default=False, help='Train with only Gauss Copulas', action='store_true')
	# TODO paths to exps

	args = parser.parse_args()

	g = 'G' if args.gauss else ''
	path_data = lambda layer: f"{conf.path2data}/outputs/{args.exp}{g}_layer{layer}.pkl"
	path_models = lambda layer: f"{conf.path2data}/outputs/{args.exp}{g}_models_layer{layer}.pkl"
	path_final = f"{conf.path2data}/outputs/{args.exp}{g}_trained.pkl"

	gpus = [0,1]
	start = time.time()
	result = train_vine(args.exp, path_data, path_models, path_final,
		layers_max=args.layers,start=args.start,gauss=args.gauss,
		device_list=[f'cuda:{i}' for i in gpus])
	end = time.time()

	print(f"Done. Training {args.start}-{len(result['models'])} trees took {(end-start)//60} min")