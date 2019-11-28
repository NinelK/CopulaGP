import pickle as pkl
import numpy as np

def load_experimental_data(path,animal,day_name,n1,n2):

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

	with open("{}/{}_{}_signals.pkl".format(path,animal,day_name),'rb') as f:
	    signal_pkl = pkl.load(f)
	with open("{}/{}_{}_behaviour.pkl".format(path,animal,day_name),'rb') as f:
	    behaviour_pkl = pkl.load(f)
	for s in ['ROIsN','trialStart','maxTrialNum','trials']:
	    assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))

	signals = signal_pkl['signals_transformed']

	data1 = data_from_n(n1)
	data2 = data_from_n(n2)

	Y_all = np.array([data1,data2]).T
	X_all = np.array(behaviour_pkl['position'])#local_time

	rule = (Y_all[:,0]>1e-4) & (Y_all[:,1]>1e-4)  \
	        & (Y_all[:,0]<1.-1e-4) & (Y_all[:,1]<1.-1e-4)
	 
	X = np.reshape(X_all[rule],(-1,1))
	X[X<0] = 160.+X[X<0]
	X[X>160] = X[X>160]-160.
	X = X/160.
	Y = Y_all[rule]
	
	return X, Y