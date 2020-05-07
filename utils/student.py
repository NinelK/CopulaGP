from scipy.special import beta, gamma, digamma
import numpy as np

# rand_u = np.linspace(-1., 1., 40)
# rand_v = np.linspace(-1., 1., 40)

# u, v = np.meshgrid(rand_u, rand_v)
# data = torch.tensor([u.flatten(),v.flatten()]).t()
# log_probs = utils.student_logprob(data,0.5,2)
# plt.imshow(log_probs.reshape(u.shape))

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = [1.]
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def det(rho,n):
    return ((n-1)*rho + 1)*(1-rho)**(n-1)

def student_H(rho,df,n):
    omega = np.log((beta(df/2,1/2)**n)*gamma(n/2)/(np.pi**(n/2) * beta(df/2,n/2))) - \
    digamma(df/2)*df*(n-1)/2 + digamma((df+1)/2)*n*(df+1)/2 - digamma((df+n)/2)*(df+n)/2
    D = det(rho,n)
    #tested on 2D
    return omega - 0.5*np.log(D)

def student_logprob(t,rho,df):
    n = t.shape[-1]
    D = det(rho,n)
    tSt = (rho * (t.sum(axis=-1))**2 - (1+(n-1)*rho)*(t**2).sum(axis=-1) )/ ((n-1)*rho**2 - (n-2)*rho-1)
    #devide by almost D, except that (1-rho)**1, not (1-rho)**(n-1)
    C = np.log(gamma(n/2)) - np.log(beta(df/2,n/2)) - 0.5*(n*np.log(np.pi*df)+np.log(D))
    return C - ((df+n)/2) * np.log(1 + tSt/df)

def student_rvs(Nvar,rhos,dfs,n):
    NSamp = rhos.shape[0]
    assert NSamp==dfs.shape[0]
    y0 = np.empty((NSamp,n,Nvar))
    for i in range(NSamp):
        rho = rhos[i]
        df = dfs[i]
        cov = np.ones(Nvar)*rho + (1-rho)*np.identity(Nvar)
        y0[i] = multivariate_t_rvs(np.zeros(Nvar), cov, df=df, n=n)
    return y0