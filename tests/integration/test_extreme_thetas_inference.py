import sys
sys.path.append('/home/nina/LFI/')
import time
import numpy as np
import torch
import gpytorch
import utils
import bvcopula
from bvcopula import conf
from numpy.testing import assert_allclose

def test_extreme_thetas_inference(X, likelihood, true_thetas, atol=0., device=torch.device('cpu')):

    t1 = time.time()

    copula_model = likelihood.copula(true_thetas) 
    Y = copula_model.sample().numpy().squeeze()
    from bvcopula import conf
    #convert numpy data to tensors (optionally on GPU)
    train_x = torch.tensor(X).float().to(device=device)
    train_y = torch.tensor(Y).float().to(device=device)
    
    _, model = bvcopula.infer([likelihood], train_x, train_y, device)
        
    model.eval()
    with torch.no_grad():
        output = model(train_x)
    gplink = model.likelihood.gplink_function
    thetas, _ = gplink(output.mean)
    thetas_max,_ = gplink(output.mean+2*output.variance)  
    thetas_min,_ = gplink(output.mean-2*output.variance)  

    t2 = time.time()

    total_time = t2-t1
    minutes = int(total_time/60)
    seconds = (int(total_time)%(60))
    print("Took {} min {} s ({})".format(minutes,seconds,int(total_time)))

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

def main():

    start = time.time()

    NSamp=2500
    X = np.linspace(0.,1.,NSamp)

    # the tests are organised in the following triplets:
    # min theta (can fail the test)
    # mid theta corresponding to maximal gradient in gp_link (can generate NaN in gradients)
    # max theta (can fail the test)

    gauss, frank, clayton, gumbel = 0,0,0,0

    gumbel+=test_extreme_thetas_inference(X, bvcopula.GumbelCopula_Likelihood(),torch.full([NSamp],conf.Gumbel_Theta_Max).float(),atol=.5)
    gumbel+=test_extreme_thetas_inference(X, bvcopula.GumbelCopula_Likelihood(),torch.full([NSamp],1.).float(),atol=.5)
    gumbel+=test_extreme_thetas_inference(X, bvcopula.GumbelCopula_Likelihood(),torch.full([NSamp],1.+np.sqrt(conf.Gumbel_Theta_Max-1.)).float(),atol=.5)

    clayton+=test_extreme_thetas_inference(X, bvcopula.ClaytonCopula_Likelihood(),torch.full([NSamp],conf.Clayton_Theta_Max).float(),atol=.5)
    clayton+=test_extreme_thetas_inference(X, bvcopula.ClaytonCopula_Likelihood(),torch.full([NSamp],0.).float(),atol=.5)
    clayton+=test_extreme_thetas_inference(X, bvcopula.ClaytonCopula_Likelihood(),torch.full([NSamp],np.sqrt(conf.Clayton_Theta_Max)-0.1).float(),atol=.5)

    frank+=test_extreme_thetas_inference(X, bvcopula.FrankCopula_Likelihood(),torch.full([NSamp],+conf.Frank_Theta_Max).float(),atol=0.7)
    frank+=test_extreme_thetas_inference(X, bvcopula.FrankCopula_Likelihood(),torch.full([NSamp],-conf.Frank_Theta_Max).float(),atol=0.7)
    frank+=test_extreme_thetas_inference(X, bvcopula.FrankCopula_Likelihood(),torch.full([NSamp],0.).float(),atol=0.1) #it's Frank copula (-inf,+inf)

    gauss+=test_extreme_thetas_inference(X, bvcopula.GaussianCopula_Likelihood(),torch.full([NSamp],1.).float(),atol=1e-4)
    gauss+=test_extreme_thetas_inference(X, bvcopula.GaussianCopula_Likelihood(),torch.full([NSamp],0.).float(),atol=0.15)
    gauss+=test_extreme_thetas_inference(X, bvcopula.GaussianCopula_Likelihood(),torch.full([NSamp],-1.).float(),atol=1e-4)

    print("Gauss ({}%), Frank ({}%), Clayton ({}%), Gumbel ({}%)".format(int(gauss*100/3),int(frank*100/3),int(clayton*100/3),int(gumbel*100/3)))

    end = time.time()

    total_time = end-start
    hours = int(total_time/60/60)
    minutes = int(total_time/60)%60
    seconds = (int(total_time)%(60))
    print("All tests took {} h {} min {} s ({})".format(hours,minutes,seconds,int(total_time)))

    assert gauss+frank+clayton+gumbel==12

if __name__ == "__main__":
    main()
