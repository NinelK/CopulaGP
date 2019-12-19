import numpy as np
from fastkde import fastKDE

def fast_signal2uniform(input,condition,numPointsPerSigma=20):
    '''
    Transforms distributions into uniform distribution
    for any value of the conditioning variable.
    
    Args:
        :attr:`input` (:class:`np.array`)
            Values of the random variable.
        :attr:`condition` (:class:`np.array`)
            Conditioning variable.
        :attr:`numPointsPerSigma` (int, default=20)
            Number of points for conditional PDF/CDF estimation.
    Returns
        :class:`np.array`
            Transformed random variable with uniform distribution
            for any given value of the conditioning variable.
    '''
    pOfYGivenX,axes = fastKDE.conditional(input,condition,numPointsPerSigma=numPointsPerSigma)

    def transform_point(x,y):
        f_x = np.sum(axes[0]<x)
        f_y = np.sum(axes[1]<y)
        wx = np.abs((axes[0][f_x]-x)/(axes[0][f_x]-axes[0][f_x-1]))
        wy = np.abs((axes[1][f_y]-y)/(axes[1][f_y]-axes[1][f_y-1]))
        return (1-wy) * ((1-wx)*cOfYGivenX[f_y,f_x] + wx*cOfYGivenX[f_y,f_x-1]) + \
                wy * ((1-wx)*cOfYGivenX[f_y-1,f_x] + wx*cOfYGivenX[f_y-1,f_x-1])

    #make CDF from PDF
    cOfYGivenX = np.empty_like(pOfYGivenX)
    for i, p in enumerate(pOfYGivenX.T):
        cOfYGivenX[:,i] = np.cumsum(p)/np.sum(p)
    #transform data points as CDF(x)
    s_tr = np.zeros_like(input)
    for i, (x,y) in enumerate(zip(condition,input)):
        s_tr[i] = transform_point(x,y)

    return s_tr

def zeroinflated_signal2uniform(input,condition,numPointsPerSigma=20):
    '''
    Transforms zero-inflated distributions into uniform distribution
    for any value of the conditioning variable.
    
    Args:
        :attr:`input` (:class:`np.array`)
            Values of the random variable with zero-inflation.
        :attr:`condition` (:class:`np.array`)
            Conditioning variable.
        :attr:`numPointsPerSigma` (int, default=20)
            Number of points for conditional PDF/CDF estimation.
    Returns
        :class:`np.array`
            Transformed random variable with uniform distribution
            for any given value of the conditioning variable.
    '''
    transformed = np.empty_like(input)
    zeros = len(input[input==0])
    part_zero = zeros/len(input)
    transformed[input!=0] = part_zero + (1-part_zero)*fast_signal2uniform(input[input!=0],condition[input!=0],numPointsPerSigma=20)
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