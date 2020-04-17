import sys
sys.path.append('/home/nina/CopulaGP/')
import time
import numpy as np
import torch
import utils
import select_copula
import bvcopula
# from select_copula import select_copula_model as select_model
from select_copula import select_with_heuristics as select_model

import pytest

def model_selection(mode, likelihoods, device=torch.device('cuda:0')):

    NSamp=5000
    X = np.linspace(0.,1.,NSamp)

    if len(likelihoods)==1:
        assert mode=='thetas'

    copula_model = utils.create_model(mode,likelihoods,X.shape[0])
    exp_name = utils.get_copula_name_string(likelihoods)

    print('Try and guess ',exp_name)

    Y = copula_model.sample().numpy().squeeze()

    train_x = torch.tensor(X).float().cuda(device=device)
    train_y = torch.tensor(Y).float().cuda(device=device)

    t1 = time.time()
    selected, _ = select_model(X,Y,device,exp_name,
                                                './temp','{}1'.format(mode[0]),'{}2'.format(mode[0]),
                                                train_x=train_x,train_y=train_y)
    t2 = time.time()
    print(f"Took {(t2-t1)//60} min")

    # assert
    print((select_copula.available_elements(likelihoods) == select_copula.available_elements(selected)))

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
                                    bvcopula.GumbelCopula_Likelihood(rotation='0°')])

        model_selection(mode, [bvcopula.FrankCopula_Likelihood(),
                                    bvcopula.GumbelCopula_Likelihood(rotation='180°')])

        model_selection(mode, [bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='270°')])

        model_selection(mode, [bvcopula.GaussianCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='0°')])

    for mode in ['mixes','thetas']:

        model_selection(mode, [bvcopula.FrankCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='0°')])

        model_selection(mode, [bvcopula.FrankCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
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
