import sys
sys.path.append('/home/nina/LFI/')
import time
import numpy as np
import torch
import gpytorch
import utils
import select_copula
import bvcopula
from bvcopula import conf
from numpy.testing import assert_allclose

import pytest

def model_selection(mode, likelihoods, device=torch.device('cpu')):

    NSamp=2000
    X = np.linspace(0.,1.,NSamp)

    if len(likelihoods)==1:
        assert mode=='thetas'

    copula_model = create_model(mode,likelihoods,X.shape[0])
    exp_name = utils.get_copula_name_string(likelihoods)

    print('Try and guess ',exp_name)

    Y = copula_model.sample().numpy().squeeze()

    train_x = torch.tensor(X).float().cuda(device=device)
    train_y = torch.tensor(Y).float().cuda(device=device)

    selected, _ = select_copula.select_copula_model(X,Y,device,exp_name,
                                                './temp','{}1'.format(mode[0]),'{}2'.format(mode[0]),
                                                train_x=train_x,train_y=train_y)

    assert (select_copula.available_elements(likelihoods) == select_copula.available_elements(selected))

    # if (select_copula.available_elements(likelihoods) == select_copula.available_elements(selected)):
    #     print('Pass. Model identified correctly!')
    #     return 0
    # else:
    #     waic, model = bvcopula.infer(likelihoods,train_x,train_y,device=device)
    #     print('Fail. WAIC of a correct model: {:.0f}'.format(waic))
    #     return 1

    # logging.info("Result: {} -> {}".format(utils.get_copula_name_string(likelihoods),utils.get_copula_name_string(selected)))

def generate_thetas(likelihoods,n):
    m = len(likelihoods)
    thetas = torch.zeros(m,n)
    pi = 2*torch.asin(torch.tensor([1]).float())
    for i, lik in enumerate(likelihoods):
        sin = torch.sin(torch.linspace(0,1,n)*2*pi + i*pi/m)
        if lik.name == 'Gaussian':
            thetas[i] = 0.+sin
        elif lik.name == 'Frank':
            thetas[i] = 0.+conf.Frank_Theta_Max*sin
        elif lik.name == 'Clayton':
            thetas[i] = conf.Clayton_Theta_Max/4*torch.abs(sin) #sin=0=independence
        elif lik.name == 'Gumbel':
            thetas[i] = 1.+(conf.Gumbel_Theta_Max/1.4-1.)/4*torch.abs(sin)
        elif lik.name == 'Independence':
            thetas[i] = 0.
        else:
            raise('Unknown copula')
            
    return thetas

def generate_mixes(likelihoods,n):
    m = len(likelihoods)
    mixes = torch.zeros(m,n)
    pi = 2*torch.asin(torch.tensor([1]).float())
    for i, lik in enumerate(likelihoods):
        mixes[i] = (1+torch.sin(torch.linspace(0,1,n)*2*pi + i*pi/m))/2
    return mixes

def basic_thetas(likelihoods,n):
    m = len(likelihoods)
    thetas = torch.zeros(m,n)
    for i, lik in enumerate(likelihoods):
        base = torch.ones(n)
        if lik.name == 'Gaussian':
            thetas[i] = 0.7*base
        elif lik.name == 'Frank':
            thetas[i] = (-2)*base
        elif lik.name == 'Clayton':
            thetas[i] = conf.Clayton_Theta_Max/4
        elif lik.name == 'Gumbel':
            thetas[i] = 1.+(conf.Gumbel_Theta_Max/1.4-1.)/4
        else:
            raise('Unknown copula')
            
    return thetas

def create_model(mode,likelihoods,n):
    m = len(likelihoods)
    copulas = [lik.copula for lik in likelihoods]
    rotations = [lik.rotation for lik in likelihoods]
    if mode=='thetas':
        copula_model = bvcopula.MixtureCopula(generate_thetas(likelihoods,n),
                    torch.ones(m,n)*(1/m),
                    copulas,
                    rotations=rotations)
    elif mode=='mixes':
        copula_model = bvcopula.MixtureCopula(basic_thetas(likelihoods,n),
                    generate_mixes(likelihoods,n),
                    copulas,
                    rotations=rotations)
    else:
        raise('Unknown model generation mode')
    return copula_model

def test_Independence():
    model_selection('thetas', [bvcopula.IndependenceCopula_Likelihood()])

def test_Gaussian():
    model_selection('thetas', [bvcopula.GaussianCopula_Likelihood()])

def test_Frank():
    model_selection('thetas', [bvcopula.FrankCopula_Likelihood()])

def test_Clayton():
    rot = np.random.choice(['0°','90°','180°','270°'])
    model_selection('thetas', [bvcopula.ClaytonCopula_Likelihood(rotation=rot)])

def test_Gumbel():
    rot = np.random.choice(['0°','90°','180°','270°'])
    model_selection('thetas', [bvcopula.GumbelCopula_Likelihood(rotation=rot)])

if __name__ == "__main__":

    start = time.time()

    model_selection('thetas', [bvcopula.GaussianCopula_Likelihood()])
    model_selection('thetas', [bvcopula.FrankCopula_Likelihood()])
    for rot in ['0°','90°','180°','270°']:
        model_selection('thetas', [bvcopula.ClaytonCopula_Likelihood(rotation=rot)])
        model_selection('thetas', [bvcopula.GumbelCopula_Likelihood(rotation=rot)])

    for mode in ['mixes','thetas']:

        model_selection(mode, [bvcopula.GumbelCopula_Likelihood(rotation='90°'),
                                    bvcopula.GaussianCopula_Likelihood()])

        model_selection(mode, [bvcopula.GaussianCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='270°')])

        model_selection(mode, [bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='270°')])

        model_selection(mode, [bvcopula.FrankCopula_Likelihood(),
                                    bvcopula.GumbelCopula_Likelihood(rotation='180°')])

        model_selection(mode, [bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='270°')])

        model_selection(mode, [bvcopula.GaussianCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='0°')])

        model_selection(mode, [bvcopula.FrankCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='0°')])

        model_selection(mode, [bvcopula.FrankCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='270°')])

        model_selection(mode, [bvcopula.GumbelCopula_Likelihood(rotation='0°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='180°'),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°')])

    end = time.time()

    total_time = end-start
    hours = int(total_time/60/60)
    minutes = int(total_time/60)%60
    seconds = (int(total_time)%(60))
    print("All tests took {} h {} min {} s ({})".format(hours,minutes,seconds,int(total_time)))
