import time
import numpy as np
import torch
import copulagp.utils as utils
import copulagp.synthetic_data as sd
import copulagp.select_copula as select_copula
import copulagp.bvcopula as bvcopula
# from copulagp.select_copula import select_copula_model as select_model
from copulagp.select_copula import select_with_heuristics as select_model
# from copulagp.select_copula import select_light as select_model

import pytest
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)

def model_selection(mode, likelihoods, device=torch.device('cuda:1')):

    NSamp=5000
    X = np.linspace(0.,1.,NSamp)

    out_path = './temp'

    if len(likelihoods)==1:
        assert mode=='thetas'

    copula_model = sd.create_model(mode,likelihoods,X.shape[0])
    exp_name = utils.get_copula_name_string(likelihoods)

    print('Try and guess ',exp_name)

    Y = copula_model.sample().numpy().squeeze()

    train_x = torch.tensor(X).float().to(device=device)
    train_y = torch.tensor(Y).float().to(device=device)

    t1 = time.time()
    selected, waic_best = select_model(X,Y,device,exp_name,
                                                out_path,f'{mode[0]}1',f'{mode[0]}2',
                                                train_x=train_x,train_y=train_y)
    t2 = time.time()
    print(f"Took {int(t2-t1)} sec")

    # assert
    selected_copula = selected.model_init(device)
    if ((select_copula.available_elements(likelihoods) == select_copula.available_elements(selected_copula.likelihood.likelihoods))):
        print('Pass')
    else:
        waic_correct, _ = bvcopula.infer(likelihoods,train_x,train_y,device=device)
        if isinstance(waic_correct,torch.Tensor):
            waic_correct = waic_correct.cpu().numpy()
        print(f'Correct WAIC: {waic_correct:.4f}, best WAIC {waic_best:.4f}, diff: {(waic_best-waic_correct):.4f}')
        if (waic_best-waic_correct)<0.05:
            print('Pass')
        else:
            print('Fail')

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
        model_selection(mode, [bvcopula.GumbelCopula_Likelihood(rotation='180°'),
                                    bvcopula.FrankCopula_Likelihood()])
        model_selection(mode, [bvcopula.ClaytonCopula_Likelihood(rotation='0°'),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°')])
        model_selection(mode, [bvcopula.ClaytonCopula_Likelihood(rotation='180°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='270°')])
    for mode in ['mixes','thetas']:

        model_selection(mode, [bvcopula.GaussianCopula_Likelihood(),
                                    bvcopula.ClaytonCopula_Likelihood(rotation='90°'),
                                    bvcopula.GumbelCopula_Likelihood(rotation='0°')])

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
    print(f"All tests took {hours} h {minutes} min {seconds} s ({int(total_time)})")
