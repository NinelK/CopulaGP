import argparse
import conf
import time
from copulagp.train import train_vine

'''
This script runs model selection for a C-Vine.
It is possible to run the model selection from a python script or a notebook as well (see Shuttle.ipynb),
although for larger models it is better to use this script. This script works with multiple GPUs and saves checkpoints regularly.

To run a model selection using this dataset.
1. Prepare a pickle file, which contains a dictionary with keys 'X' and 'Y'.
	'X' - conditioning variable, normalised to [0,1] (nd.array of size N)
	'Y' - an numpy array of D variables (np.array of size N x D)
	name it following this pattern: "datasetname_layer0.pkl"
	Also, if using any flags, append them to the end of the dataset name, e.g. "datasetnameGL_layer0.pkl"
	(this naming convention is defined by this script and can be easily changed)
2. Configure the conf.py in the main folder. Provide a path to datasets and a path for outputs.
3. If you use multiple GPUs, provide a list of device numbers in this script (gpus)
4. Run "python train.py -exp datasetname"

'''

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train copula vine model')
	parser.add_argument('-exp', default='', help='Experiment name')
	parser.add_argument('-layers', default=-1, help='How many layers? (-1 = all possible)', type=int)
	parser.add_argument('-start', default=0, help='Re-Start from a certain layer', type=int)
	parser.add_argument('--gauss','-g', default=False, help='Train with only Gauss Copulas', action='store_true')
	parser.add_argument('--light','-l', default=False, help='Light model selection, without Gumbel', action='store_true')
	parser.add_argument('--shuffle','-s', default=False, help='Shuffle X', action='store_true')
	# TODO paths to exps

	args = parser.parse_args()

	g = '_G' if args.gauss else ''
	if args.light:
		g += 'L'
	if args.shuffle:
		g += 'S'
	print(g)
	path_data = lambda layer: f"{conf.path2data}/{args.exp}{g}_layer{layer}.pkl"
	path_models = lambda layer: f"{conf.path2outputs}/{args.exp}{g}_models_layer{layer}.pkl"
	path_final = f"{conf.path2outputs}/{args.exp}{g}_trained.pkl"
	path_logs = lambda exp_pref, layer: f'{conf.path2outputs}/logs_{exp_pref}/layer{layer}'

	gpus = range(2,8)
	start = time.time()
	result = train_vine(args.exp, path_data, path_models, path_final,
		layers_max=args.layers,start=args.start,gauss=args.gauss,
		light=args.light,
		shuffle=args.shuffle,
		path_logs=path_logs,
		device_list=[f'cuda:{i}' for i in gpus])
	end = time.time()

	print(f"Done. Training {args.start}-{len(result['models'])} trees took {(end-start)//60} min")
