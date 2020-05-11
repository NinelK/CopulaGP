import numpy as np
from math import log,pi,exp

#Auxilary functions
def vd(d,q):
    # Compute the volume of unit l_q ball in d dimensional space
    if (q==float('inf')):
        return d*np.log(2)
    return d*np.log(2*gamma(1+1.0/q)) - np.log(gamma(1+d*1.0/q))

def BI_KSG(x,y,k=5,q=float('inf')):
    '''
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)
        Input: x: 2D list of size N*d_x
        y: 2D list of size N*d_y
        k: k-nearest neighbor parameter
        q: l_q norm used to decide k-nearest distance
        Output: one number of I(X;Y), entropy of Y
    '''

    import scipy.spatial as ss
    from scipy.special import digamma,gamma

    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    dx = len(x[0])
    dy = len(y[0])
    data = np.concatenate((x,y),axis=1)

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=q)[0][k] for point in data]
    ans_xy = -digamma(k) + np.log(N) + vd(dx+dy,q)
    ans_x = np.log(N) + vd(dx,q)
    ans_y = np.log(N) + vd(dy,q)
    for i in range(N):
        ans_xy += (dx+dy)*np.log(knn_dis[i])/N
        ans_x += -np.log(len(tree_x.query_ball_point(x[i],knn_dis[i]+1e-15,p=q))-1)/N+dx*np.log(knn_dis[i])/N
        ans_y += -np.log(len(tree_y.query_ball_point(y[i],knn_dis[i]+1e-15,p=q))-1)/N+dy*np.log(knn_dis[i])/N		
    return (ans_x+ans_y-ans_xy)/np.log(2), (ans_xy-ans_x)/np.log(2)

def LNN_entropy(x,k=5,tr=30,bw=0):

	'''
		Estimate the entropy H(X) from samples {x_i}_{i=1}^N
		Using Local Nearest Neighbor (LNN) estimator with order 2

		Input: x: 2D list of size N*d_x
		k: k-nearest neighbor parameter
		tr: number of sample used for computation
		bw: option for bandwidth choice, 0 = kNN bandwidth, otherwise you can specify the bandwidth

		Output: one number of H(X)
	'''
	import scipy.spatial as ss
	from scipy.special import digamma,gamma

	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	assert tr <= len(x)-1, "Set tr smaller than num.samples - 1"
	N = len(x)
	d = len(x[0])
	
	local_est = np.zeros(N)
	S_0 = np.zeros(N)
	S_1 = np.zeros(N)
	S_2 = np.zeros(N)
	tree = ss.cKDTree(x)
	if (bw == 0):
		bw = np.zeros(N)
	for i in range(N):
		lists = tree.query(x[i],tr+1,p=2)
		knn_dis = lists[0][k]
		list_knn = lists[1][1:tr+1]
		
		if (bw[i] == 0):
			bw[i] = knn_dis

		S0 = 0
		S1 = np.matrix(np.zeros(d))
		S2 = np.matrix(np.zeros((d,d)))
		for neighbor in list_knn:
			dis = np.matrix(x[neighbor] - x[i])
			S0 += exp(-dis*dis.transpose()/(2*bw[i]**2))
			S1 += (dis/bw[i])*exp(-dis*dis.transpose()/(2*bw[i]**2))
			S2 += (dis.transpose()*dis/(bw[i]**2))*exp(-dis*dis.transpose()/(2*bw[i]**2))

		Sigma = S2/S0 - S1.transpose()*S1/(S0**2)
		det_Sigma = np.linalg.det(Sigma)
		if (det_Sigma < (1e-4)**d):
			local_est[i] = 0
		else:
			offset = (S1/S0)*np.linalg.inv(Sigma)*(S1/S0).transpose()
			local_est[i] = -log(S0) + log(N-1) + 0.5*d*log(2*pi) + d*log(bw[i]) + 0.5*log(det_Sigma) + 0.5*offset[0][0]

	if (np.count_nonzero(local_est) == 0):
		return 0
	else: 
		return np.mean(local_est[np.nonzero(local_est)])

def LNN_MI(x,y,k=5,tr=30):
	'''
		Estimate the mutual information I(X;Y) from samples {x_i,y_i}_{i=1}^N
		Using I(X;Y) = H_{LNN}(X) + H_{LNN}(Y) - H_{LNN}(X;Y)
		where H_{LNN} is the LNN entropy estimator with order 2

		Input: x: 2D list of size N*d_x
		y: 2D list of size N*d_y
		k: k-nearest neighbor parameter
		tr: number of sample used for computation

		Output: one number of I(X;Y)
	'''
	assert len(x)==len(y), "x and y must have the same number of samples"
	assert len(x[0])>=1

	ans_x = LNN_entropy(x,k,tr)
	ans_y = LNN_entropy(y,k,tr) 
	ans_xy = LNN_entropy(np.concatenate([x,y],axis=1),k,tr)
	return ans_x + ans_y - ans_xy, ans_y

def Mixed_KSG(x,y,k=5):
	'''
		Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
		Using *Mixed-KSG* mutual information estimator
		Input: x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
		y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
		k: k-nearest neighbor parameter
		Output: one number of I(X;Y)
	'''

	import scipy.spatial as ss
	from scipy.special import digamma

	assert len(x)==len(y), "Lists should have same length"
	assert k <= len(x)-1, "Set k smaller than num. samples - 1"
	N = len(x)
	if x.ndim == 1:
		x = x.reshape((N,1))
	dx = len(x[0])	
	if y.ndim == 1:
		y = y.reshape((N,1))
	dy = len(y[0])
	data = np.concatenate((x,y),axis=1)

	tree_xy = ss.cKDTree(data)
	tree_x = ss.cKDTree(x)
	tree_y = ss.cKDTree(y)

	knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
	ans, ans_x, ans_y, ans_xy = 0, 0, 0, 0

	for i in range(N):
		kp, nx, ny = k, k, k
		if knn_dis[i] == 0:
			print('Wow, discrete!')
			kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
			nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
		else:
			nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
			ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
		# ans += (digamma(kp) + log(N) - digamma(nx) - digamma(ny))/N
		ans_xy += (-digamma(kp) + log(N) + (dx+dy)*np.log(knn_dis[i]))/N
		ans_x += (-digamma(nx) + log(N)+ dx*np.log(knn_dis[i]))/N
		ans_y += (-digamma(ny) + log(N)+ dy*np.log(knn_dis[i]))/N
	return (ans_x+ans_y-ans_xy)/np.log(2), (ans_xy-ans_x)/np.log(2)