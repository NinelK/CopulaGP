import numpy as np

from torch import Size
def binned_MI(vine, half_bin=200, step=100, sample_size=Size([10])):
    '''
    Analytically estimates MI between inputs and variables
    (sampled from vine copula model).
    Uses Pearson's correlation coefficient on binarized data. 
    Only works for 2D variables. 
    '''
    import numpy as np
    N = vine.inputs.numel()
    y = vine.sample(sample_size).cpu().numpy()
    p = np.zeros((N//step,2))
    for j in range(len(p)):
        start = np.clip(step * j - half_bin,0,N)
        end = np.clip(step * j + half_bin,0,N)
        p[j,0] = np.corrcoef(*y[start:end].reshape(-1,y.shape[-1]).T)[0,1]
        p[j,1] = end-start
    assert end==N
    Hs = np.sum((-np.log(1-p[:,0]**2)/2/np.log(2))*p[:,1]/p[:,1].sum())
    p_all = np.corrcoef(*y.reshape(-1,y.shape[-1]).T)[0,1]
    H = (-np.log(1-p_all**2)/2/np.log(2))
    return (Hs-H, H)
