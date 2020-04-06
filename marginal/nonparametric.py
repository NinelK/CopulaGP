import numpy as np
from fastkde import fastKDE
import scipy.interpolate as intrp

def interpolate2d_masked_array(masked_array2d):
    m = ~masked_array2d.mask.flatten()
    X,Y = masked_array2d.shape
    xx,yy = np.mgrid[:X,:Y]
    x = np.unique(xx.flatten()[m])
    y = np.unique(yy.flatten()[m])
    sel = np.ix_(x,y)
    if any(masked_array2d.mask[sel].flatten()):
        raise ValueError("Points are not on the grid")
    return intrp.RectBivariateSpline(x,y,masked_array2d.data[sel], kx=1, ky=1)

def transform_coord(x,y,axes):
    f_x = np.sum(axes[0]<x)
    f_y = np.sum(axes[1]<y)
    wx = (axes[0][f_x]-x)/(axes[0][f_x]-axes[0][f_x-1])
    wy = (axes[1][f_y]-y)/(axes[1][f_y]-axes[1][f_y-1])
    return (f_y - wy),(f_x - wx)

def fast_signal2uniform(input,condition,numPointsPerSigma=20):

    pOfYGivenX,axes = fastKDE.conditional(input,condition,numPointsPerSigma=numPointsPerSigma)

    #make CDF from PDF
    cOfYGivenX = np.empty_like(pOfYGivenX)
    for i, p in enumerate(pOfYGivenX.T):
        cOfYGivenX[:,i] = np.cumsum(p)/np.sum(p)
        
    f = interpolate2d_masked_array(cOfYGivenX)
    #transform data points as CDF(x)
    s_tr = np.zeros_like(input)
    for i, (x,y) in enumerate(zip(condition,input)):
        s_tr[i] = f(*transform_coord(x,y,axes))

    return s_tr

def zeroinflated_signal2uniform(input,condition,numPointsPerSigma=50):

    transformed = np.empty_like(input)
    zeros = len(input[input==0])
    part_zero = zeros/len(input)
    transformed[input!=0] = part_zero + (1-part_zero)*fast_signal2uniform(input[input!=0],condition[input!=0],numPointsPerSigma=numPointsPerSigma)
    transformed[input==0] = np.random.rand(zeros)*part_zero

    return transformed

def single_unit_MI(response,stimulus):
    '''
    Estimates mutual information (MI) between a single variable and a conditioning variable:
    .. math::
        I(S;R) = \sum\limits_{r,s} P(r|s) P(s)\, \mathrm{log_2}\frac{P(r|s)}{P(r)}
    Args:
        :attr:`response` (:class:`np.array`)
            Values of :math:`R`.
        :attr:`stimulus` (:class:`np.array`)
            Values of :math:`S`.
        :attr:`kwargs`
    Returns
        `real` (mutual information in bits)
    '''
    pOfYGivenX,axes = fastKDE.conditional(response,stimulus,numPointsPerSigma=30)
    working_range = [np.sum(axes[0]<np.min(stimulus)),np.sum(axes[0]<np.max(stimulus))]
    assert np.all(pOfYGivenX[:,working_range[0]:working_range[1]].mask==False)
    Ns = pOfYGivenX[:,working_range[0]:working_range[1]].shape[1]
    pRS = pOfYGivenX[:,working_range[0]:working_range[1]]
    pR = np.sum(pRS,axis=1)*1/Ns
    I=0
    for i in range(Ns):
        I+=np.sum(pRS[:,i]*1/Ns*np.log(pRS[:,i]/pR)/np.log(2))
    return I
