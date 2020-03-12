import scipy.spatial as ss
from scipy.special import digamma,gamma
import numpy as np

#Auxilary functions
def vd(d,q):
    # Compute the volume of unit l_q ball in d dimensional space
    if (q==float('inf')):
        return d*np.log(2)
    return d*np.log(2*gamma(1+1.0/q)) - np.log(gamma(1+d*1.0/q))

def revised_mi(x,y,k=5,q=float('inf')):
    '''
        Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
        Using *REVISED* KSG mutual information estimator (see arxiv.org/abs/1604.03006)
        Input: x: 2D list of size N*d_x
        y: 2D list of size N*d_y
        k: k-nearest neighbor parameter
        q: l_q norm used to decide k-nearest distance
        Output: one number of I(X;Y), entropy of Y
    '''

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
    return ans_x+ans_y-ans_xy