from scipy.stats import kendalltau
import numpy as np
import pickle as pkl
from copulagp.utils import load_samples

def heuristic_element_order(samples):
    '''
    FROM MIXEDVINE
    Finds an order of elements that heuristically facilitates vine
    modelling.  For this purpose, Kendall's tau is calculated between
    samples of pairs of elements and elements are scored according to the
    sum of absolute Kendall's taus of pairs the elements appear in.
    Parameters
    ----------
    samples : array_like
        n-by-d matrix of samples where n is the number of samples and d is
        the number of marginals.
    Returns
    -------
    order : array_like
        Permutation of all element indices reflecting descending scores.
    '''

    dim = samples.shape[1]
    # Score elements according to total absolute Kendall's tau
    score = np.zeros(dim)
    for i in range(1, dim):
        for j in range(i):
            tau, _ = kendalltau(samples[:, i], samples[:, j])
            if not np.isnan(tau):
                score[i] += np.abs(tau)
                score[j] += np.abs(tau)
    #print(score)
    # Get order indices for descending score
    order = score.argsort()[::-1]
    return order

if __name__ == "__main__":
    
    animal = sys.argv[1]#'ST263'
    dayN = sys.argv[2]
    day_name = f"Day{dayN}"
    exp_pref = f"{animal}_{day_name}"

    path = '../../VRData/Processing/pkls'
    path_models = '../models'

    #samples = utils.load_neurons_only(path,exp_pref) #load only neurons
    samples = utils.load_samples(path,exp_pref)

    order = heuristic_element_order(samples)

    with open(f"{path_models}/order_{exp_pref}.pkl","wb") as f:
        pkl.dump(order,f)

    print(order)


