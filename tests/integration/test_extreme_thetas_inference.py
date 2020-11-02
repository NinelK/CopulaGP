import sys
sys.path.append('/home/nina/CopulaGP/')
import time
import numpy as np
import torch
import gpytorch
import utils
import bvcopula
import pytest
from bvcopula import conf
from numpy.testing import assert_allclose

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
np.random.seed(0)

def extreme_thetas_inference(X, bvc, true_thetas, atol=0., device=torch.device('cpu')):

    t1 = time.time()

    copula_model = bvc.copula(true_thetas) 
    Y = copula_model.sample().numpy().squeeze()
    from bvcopula import conf
    #convert numpy data to tensors (optionally on GPU)
    train_x = torch.tensor(X).float().to(device=device)
    train_y = torch.tensor(Y).float().to(device=device)
    
    _, model = bvcopula.infer([bvc], train_x, train_y, device)
        
    model.gp_model.eval()
    with torch.no_grad():
        output = model.gp_model(train_x)
    thetas, _ = model.gplink(output.mean)
    thetas_max,_ = model.gplink(output.mean+2*output.variance)  
    thetas_min,_ = model.gplink(output.mean-2*output.variance)  

    t2 = time.time()

    total_time = t2-t1
    minutes = int(total_time/60)
    seconds = (int(total_time)%(60))
    print(f"Took {minutes} min {seconds} s ({int(total_time)})")

    cpu_thetas = thetas.cpu().squeeze()
    #max_diff = np.abs((cpu_thetas-true_thetas).numpy()).max()
    #print("Max diff: {}, rel: {}".format(max_diff,max_diff/true_thetas.mean()))

    print("Thetas mean: {:.2f} (correct: {:.2f}) and std: {:.2f}".format(cpu_thetas.numpy().mean(),
        true_thetas.numpy().mean(),cpu_thetas.numpy().std()))

    if torch.all(thetas_min.cpu().squeeze() <= true_thetas+atol) & torch.all(true_thetas-atol <= thetas_max.cpu().squeeze()):
        res = "Pass"
    else:
        res = "Fail"
        print(thetas_min.cpu().max(),true_thetas.mean(),thetas_max.cpu().min())

    print("Correct value lays in ({:.2f},{:.2f}) everywhere? {}".format(thetas_min.min(),thetas_max.max(),res))

    # assert_allclose(cpu_thetas.numpy(),true_thetas,rtol=rtol,atol=atol)

    if res=='Pass':
        return 1
    else:
        return 0

def try_copula(bvc, minTh, midTh, maxTh, 
                ex_atol, mid_atol):

    NSamp=2500
    X = np.linspace(0.,1.,NSamp)

    # the tests are organised in the following triplets:
    # min theta (can fail the test)
    # mid theta corresponding to maximal gradient in gp_link (can generate NaN in gradients)
    # max theta (can fail the test)

    success = 0
    success+=extreme_thetas_inference(X, bvc, torch.full([NSamp],minTh).float(),atol=ex_atol)
    success+=extreme_thetas_inference(X, bvc, torch.full([NSamp],midTh).float(),atol=mid_atol)
    success+=extreme_thetas_inference(X, bvc, torch.full([NSamp],maxTh).float(),atol=ex_atol)

    assert success==3

def test_gumbel():
    try_copula(bvcopula.GumbelCopula_Likelihood(),1.,1.+np.sqrt(conf.Gumbel_Theta_Max-1.),conf.Gumbel_Theta_Max,
                .7, .7)

def test_clayton():
    try_copula(bvcopula.ClaytonCopula_Likelihood(),0.,np.sqrt(conf.Clayton_Theta_Max)-0.1,conf.Clayton_Theta_Max,
                .7, .7)

def test_gauss():
    try_copula(bvcopula.GaussianCopula_Likelihood(),-1.,0.,1.,
                1e-4, .15)

def test_frank():
    try_copula(bvcopula.FrankCopula_Likelihood(),-conf.Frank_Theta_Max,0.,conf.Frank_Theta_Max,
                1., .1)

if __name__ == "__main__":

    start = time.time()

    test_gumbel()
    test_clayton()
    test_frank()
    test_gauss()

    end = time.time()

    total_time = end-start
    hours = int(total_time/60/60)
    minutes = int(total_time/60)%60
    seconds = (int(total_time)%(60))
    print(f"All tests took {hours} h {minutes} min {seconds} s ({int(total_time)})")

