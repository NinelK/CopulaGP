import numpy as np
def load(c=0):
	'''
	Loads UCL shuttle data and splits it into 
	trian and test sets.
	Parameters
	----------
	c: int (default=0)
		Conditional variable index.
		Condition on time by default.
	'''
	with open("shuttle.trn","r") as f:
	    all_lines = f.read()
	    
	lines = all_lines.split('\n')
	train = np.array([[float(n) for n in l.split(' ')] for l in lines[:-1]])

	with open("shuttle.tst","r") as f:
	    all_lines = f.read()
	    
	lines = all_lines.split('\n')
	test = np.array([[float(n) for n in l.split(' ')] for l in lines[:-1]])

	# now, use train & test to estimate
	# an empirical CDF and transform the data
	size = train.shape[0]
	numbers = np.concatenate([train,test])
	numbers[:,c] = (np.argsort(numbers[:,c].flatten()).argsort()/numbers[:,c].size)\
					.reshape(numbers[:,c].shape)
	train = numbers[:size]
	test = numbers[size:]

	return (train,test)