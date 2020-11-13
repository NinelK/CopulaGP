import sys
home = '/home/nina/CopulaGP/'
sys.path.insert(0, home)

import torch
from vine import CVine
import bvcopula
import utils
import numpy as np
from scipy.stats import norm

def train4entropy(x,y,likelihood,device=torch.device('cpu'),mc_size=2500,shuffle=False,sem_tol=0.01,v=False):
    NSamp = y.shape[0]
    assert x.shape[0]==NSamp
    Nvar = y.shape[-1]
    data_layers = [torch.tensor(y).clamp(0.001,0.999).float().to(device)]
    copula_layers = []
    for m in range(0,Nvar-1):
        copulas, layer = [], []
        for n in range(1,Nvar-m):
            if shuffle:
                x = x[torch.randperm(NSamp)]
            samples = data_layers[-1][...,[n,0]]
            waic, model = bvcopula.infer(likelihood,x,samples,device=device) 
            print(f'{m},{n+m} WAIC: {waic:.3f}')
            if shuffle:
                x = x[torch.randperm(NSamp)]
            with torch.no_grad():
                f = model.gp_model(x).mean
                copula = model.likelihood.get_copula(f)
                copulas.append(copula)
                layer.append(copula.ccdf(samples))
        data_layers.append(torch.stack(layer,dim=-1))
        copula_layers.append(copulas)
    print('Trained')
    vine_trained = CVine(copula_layers,x,device=device)
    entropies = vine_trained.entropy(sem_tol=sem_tol, mc_size=mc_size, v=v)
    return vine_trained, entropies

def integrate_student(Nvar,Frhos,Fdfs,sem_tol=0.01,verbose=False):
    NN=1000 #number of inputs
    N=10 #number of samples for each input
    M=1000 #number of permutations
    rhos = Frhos(NN)
    dfs = Fdfs(NN)
    sem = float("inf")
    HRS,HR, var_sum, k = 0,0,0,0
    while sem>sem_tol:
        samples = utils.student_rvs(Nvar,rhos,dfs,N)
        samples = np.einsum("ij...->ji...",samples)
        log_probs = utils.student_logprob(samples,rhos,dfs)
        Hrs = log_probs.mean(axis=-1)
        Hr = np.zeros((N,NN))
        v = np.zeros((N,NN))
        tol = float("inf")
        kR=0
        while tol>sem_tol:
            new_Hr = np.zeros((M,N,NN))
            for l in range(M):
                perm = np.random.permutation(np.arange(len(rhos)))
                new_Hr[l] = np.exp(utils.student_logprob(samples,rhos[perm],dfs[perm]))
            kR+=1
            Hr += (new_Hr.mean(axis=0) - Hr)/kR
            v += ((new_Hr-Hr)**2).sum(axis=0)
            sR = norm.ppf(1 - 0.05) * (v / (kR * M * (kR * M - 1)))**(.5)
            tol = (sR/Hr.mean(axis=0)).max()
            # if verbose & (kR%10==0):
            #     print(tol)
        Hr = np.log(Hr).mean(axis=-1)
        
        k+=1
        HR += (Hr.mean()-HR)/k
        HRS += (Hrs.mean()-HRS)/k
        assert len(Hr)==N
        var_sum += ((Hr-HR)**2).sum() + ((Hrs-HRS)**2).sum()
        sem = norm.ppf(1 - 0.05) * (var_sum / (k * N * (k * N - 1)))**(.5)
        if verbose:
            print(HRS-HR,sem, kR)
    return ((HRS-HR,HRS,sem)/np.log(2))
