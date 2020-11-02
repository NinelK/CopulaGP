from . import BI_KSG

def HYgX(X,Y):
	'''
	Calculate conditional entropy with BI KSG method
	'''
	assert X.shape == Y[...,0].shape
	Hs = []
	for i in range(1,9):
	    _,H = BI_KSG(X.reshape(-1,1),Y[:,i].reshape(-1,1))
	    Hs.append(H)
	return Hs

	# 'numbers[:,0].reshape(-1,1)'