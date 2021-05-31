from . import BI_KSG

def HYgX(X,Y):
	'''
	Calculate conditional entropy with BI KSG method
	'''
	assert X.shape == Y[...,0].shape
	Hs = []
	for y in Y.T:
	    _,H = BI_KSG(X.reshape(-1,1),y.reshape(-1,1))
	    Hs.append(H)
	return Hs

	# 'numbers[:,0].reshape(-1,1)'