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

def test_extreme_thetas_inference(X, likelihood, true_thetas, atol=0.):

    t1 = time.time()

    copula_model = likelihood.copula(true_thetas) 
    Y = copula_model.sample().numpy().squeeze()
    from bvcopula import conf
    #convert numpy data to tensors (optionally on GPU)
    train_x = torch.tensor(X).float().cuda(device=0)
    train_y = torch.tensor(Y).float().cuda(device=0)
    # define the model (optionally on GPU)
    grid_size = 128
    model = bvcopula.Mixed_GPInferenceModel(bvcopula.MixtureCopula_Likelihood([likelihood]), 
                                            1,
                                            prior_rbf_length=0.5, 
                                            grid_size=grid_size).cuda(device=0)


    # train the model

    optimizer = torch.optim.Adam([
        {'params': model.mean_module.parameters()},
        {'params': model.variational_strategy.parameters()},
        {'params': model.covar_module.parameters(), 'lr': conf.hyper_lr}, #hyperparameters
    ], lr=conf.base_lr)

    mll = utils.VariationalELBO(model.likelihood, model, torch.ones_like(train_x), 
                                num_data=train_y.size(0), particles=torch.Size([0]), combine_terms=True)

    losses = []

    model.train()
    p = 0.
    half_iter_print = int(conf.iter_print/2)
    for i in range(conf.max_num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        #with gpytorch.settings.num_gauss_hermite_locs(50): 
        loss = -mll(output, train_y)  
        if len(losses)>conf.iter_print: 
            p += np.abs(np.mean(losses[-half_iter_print:]) - np.mean(losses[-conf.iter_print:-half_iter_print]))
        losses.append(loss.detach().cpu().numpy())
        if not (i + 1) % conf.iter_print:
            mean_p = p/conf.iter_print       
            if (0 < mean_p < conf.loss_tol):
                print("{} (theta={:.3}) converged in {} steps!".format(likelihood.name,true_thetas.mean(),i+1))
                break
            p = 0.
        # The actual optimization step
        loss.backward()
        covar_grad = model.variational_strategy.variational_distribution.chol_variational_covar.grad
        # strict
        if not torch.all(covar_grad==covar_grad):
            print(model.variational_strategy.variational_distribution.variational_mean.mean())
        assert torch.all(covar_grad==covar_grad)
        optimizer.step()
    if mean_p >= conf.loss_tol:
        print("{} (theta={:.3}) did not converge in {} steps.".format(likelihood.name,true_thetas.mean(),conf.max_num_iter))
        
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

    print("Thetas mean: {:.3} and std: {:.3}".format(cpu_thetas.numpy().mean(),cpu_thetas.numpy().std()))

    if torch.all(thetas_min.cpu().squeeze() <= true_thetas+atol) & torch.all(true_thetas-atol <= thetas_max.cpu().squeeze()):
        res = "Pass"
    else:
        res = "Fail"
        print(thetas_min.cpu().max(),true_thetas.mean(),thetas_max.cpu().min())

    print("Correct value lays in ({:.3},{:.3}) everywhere? {}".format(thetas_min.min(),thetas_max.max(),res))

    # assert_allclose(cpu_thetas.numpy(),true_thetas,rtol=rtol,atol=atol)

    if res=='Pass':
        return 1
    else:
        return 0

def main():

    start = time.time()

    NSamp=10000
    X = np.linspace(0.,1.,NSamp)

    # the tests are organised in the following triplets:
    # min theta (can fail the test)
    # mid theta corresponding to maximal gradient in gp_link (can generate NaN in gradients)
    # max theta (can fail the test)

    gauss, frank, clayton, gumbel = 0,0,0,0

    gumbel+=test_extreme_thetas_inference(X, bvcopula.GumbelCopula_Likelihood(),torch.full([NSamp],conf.Gumbel_Theta_Max).float(),atol=.2)
    gumbel+=test_extreme_thetas_inference(X, bvcopula.GumbelCopula_Likelihood(),torch.full([NSamp],1.).float(),atol=.2)
    gumbel+=test_extreme_thetas_inference(X, bvcopula.GumbelCopula_Likelihood(),torch.full([NSamp],1.+np.sqrt(conf.Gumbel_Theta_Max-1.)).float(),atol=.2)

    clayton+=test_extreme_thetas_inference(X, bvcopula.ClaytonCopula_Likelihood(),torch.full([NSamp],conf.Clayton_Theta_Max).float(),atol=.5)
    clayton+=test_extreme_thetas_inference(X, bvcopula.ClaytonCopula_Likelihood(),torch.full([NSamp],0.).float(),atol=.5)
    clayton+=test_extreme_thetas_inference(X, bvcopula.ClaytonCopula_Likelihood(),torch.full([NSamp],np.sqrt(conf.Clayton_Theta_Max)-0.1).float(),atol=.5)

    frank+=test_extreme_thetas_inference(X, bvcopula.FrankCopula_Likelihood(),torch.full([NSamp],+conf.Frank_Theta_Max).float(),atol=0.7)
    frank+=test_extreme_thetas_inference(X, bvcopula.FrankCopula_Likelihood(),torch.full([NSamp],-conf.Frank_Theta_Max).float(),atol=0.7)
    frank+=test_extreme_thetas_inference(X, bvcopula.FrankCopula_Likelihood(),torch.full([NSamp],0.).float(),atol=0.1) #it's Frank copula (-inf,+inf)

    gauss+=test_extreme_thetas_inference(X, bvcopula.GaussianCopula_Likelihood(),torch.full([NSamp],1.).float(),atol=1e-4)
    gauss+=test_extreme_thetas_inference(X, bvcopula.GaussianCopula_Likelihood(),torch.full([NSamp],0.).float(),atol=0.07)
    gauss+=test_extreme_thetas_inference(X, bvcopula.GaussianCopula_Likelihood(),torch.full([NSamp],-1.).float(),atol=1e-4)

    print("Gauss ({}%), Frank ({}%), Clayton ({}%), Gumbel ({}%)".format(int(gauss*100/3),int(frank*100/3),int(clayton*100/3),int(gumbel*100/3)))

    end = time.time()

    total_time = end-start
    hours = int(total_time/60/60)
    minutes = int(total_time/60)%60
    seconds = (int(total_time)%(60))
    print("All tests took {} h {} min {} s ({})".format(hours,minutes,seconds,int(total_time)))

if __name__ == "__main__":
    main()
