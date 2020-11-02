import pickle as pkl
import numpy as np

def _validate_data(data):
	assert data['X'].shape[0]==data['Y'].shape[0]

	assert (data['X'].min()>=0) & (data['X'].max()<=1)
	assert (data['Y'].min()>0) & (data['Y'].max()<1)

def standard_loader(path,n1=None,n2=None):
	'''
	The simplest loader, that expects to get a pkl file
	as input, with a dictionary containing all variables
	(Y) and a normalised conditioning variable (X).
	'''
	with open(path,"rb") as f:
		data = pkl.load(f)
	
	assert 'X' in data.keys()
	assert 'Y' in data.keys()

	_validate_data(data)

	if n1 is None:
		return data['X'], data['Y']
	else:
		assert n2 is not None
		assert max(n1,n2)<data['Y'].shape[1]
		return data['X'], data['Y'][:,[n1,n2]]

def standard_saver(path, x, y):
	# save layer
	assert np.all(y>=0) & np.all(y<=1)

	data = {}
	data['X'] = x
	data['Y'] = y
	
	_validate_data(data)

	with open(path,"wb") as f:
	    pkl.dump(data,f)

def load_experimental_data(path,animal,day_name,n1,n2):
	'''
		Loads experimental data
		TODO: squash parameters into f"{path}/{animal}_{day_name}"
	'''
	def data_from_n(n):
		if n>=0:
			data = signals[n]
		elif n==-1:
			data = behaviour_pkl['transformed_velocity']
		elif n==-2:
			data = behaviour_pkl['transformed_licks']
		elif n==-3:
			data = (behaviour_pkl['transformed_early_reward'] + behaviour_pkl['transformed_late_reward'])/2
		elif n==-4:
			data = behaviour_pkl['transformed_early_reward']
		elif n==-5:
			data = behaviour_pkl['transformed_late_reward']
		else:
			raise ValueError('n is out of range')
		return data

	exp_pref = f"{animal}_{day_name}"
	with open(f"{path}/{exp_pref}_signals.pkl",'rb') as f:
		signal_pkl = pkl.load(f)
	with open(f"{path}/{exp_pref}_behaviour.pkl",'rb') as f:
		behaviour_pkl = pkl.load(f)
	for s in ['ROIsN','trialStart','maxTrialNum','trials']:
		assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))

	signals = signal_pkl['signals_transformed']

	data1 = data_from_n(n1)
	data2 = data_from_n(n2)

	Y_all = np.array([data1,data2]).T
	X_all = np.array(behaviour_pkl['position'])#local_time
	 
	X = (X_all%160)/160
	Y = (Y_all - Y_all.min(axis=0))/(Y_all.max(axis=0)-Y_all.min(axis=0))*0.998+0.001
	# constrain all Ys between 0.01 and 0.999
	
	return X, Y

def load_transformed_samples(path,exp_pref):
    with open(f"{path}/{exp_pref}_signals.pkl",'rb') as f:
        signal_pkl = pkl.load(f)
    with open(f"{path}/{exp_pref}_behaviour.pkl",'rb') as f:
        behaviour_pkl = pkl.load(f)
    for s in ['ROIsN','trialStart','maxTrialNum','trials']:
        assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))
    #stimulus = (behaviour_pkl['position']/160)%1

    samples = np.zeros((signal_pkl['signals_fissa'].shape[1], #n
        signal_pkl['signals_fissa'].shape[0]+5),dtype=float) #d

    samples[:,0] = behaviour_pkl['transformed_late_reward']
    samples[:,1] = behaviour_pkl['transformed_early_reward']
    samples[:,2] = behaviour_pkl['transformed_early_reward'] + behaviour_pkl['transformed_late_reward']
    samples[:,3] = behaviour_pkl['transformed_licks']
    samples[:,4] = behaviour_pkl['transformed_velocity']

    samples[:,5:] = signal_pkl['signals_transformed'].T

    return samples

def load_samples(path,exp_pref):
    with open(f"{path}/{exp_pref}_signals.pkl",'rb') as f:
        signal_pkl = pkl.load(f)
    with open(f"{path}/{exp_pref}_behaviour.pkl",'rb') as f:
        behaviour_pkl = pkl.load(f)
    for s in ['ROIsN','trialStart','maxTrialNum','trials']:
        assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))
    #stimulus = (behaviour_pkl['position']/160)%1

    samples = np.zeros((signal_pkl['signals_fissa'].shape[1], #n
        signal_pkl['signals_fissa'].shape[0]+5),dtype=float) #d

    samples[:,0] = behaviour_pkl['fat_late_reward']
    samples[:,1] = behaviour_pkl['fat_early_reward']
    samples[:,2] = behaviour_pkl['fat_early_reward'] + behaviour_pkl['fat_late_reward']
    samples[:,3] = behaviour_pkl['fat_licks']
    samples[:,4] = behaviour_pkl['velocity']

    samples[:,5:] = signal_pkl['signals_fissa'].T

    return samples

def load_neurons_only(path,exp_pref):
    with open(f"{path}/{exp_pref}_signals.pkl",'rb') as f:
        signal_pkl = pkl.load(f)

    return signal_pkl['signals_fissa'].T