import torch
from . import conf
import logging
import bvcopula
from utils import get_copula_name_string, Plot_Fit
import os

from .importance import important_copulas, reduce_model
   
def select_with_heuristics(X: torch.Tensor, Y: torch.Tensor, device: torch.device,
    exp_pref: str, path_output: str, name_x: str, name_y: str,
    train_x = None, train_y = None):

    exp_name = f'{exp_pref}_{name_x}-{name_y}'
    log_name = f'{path_output}/log_{device}_{exp_name}.txt'
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.basicConfig(filename=log_name, filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')

    logging.info(f'Selecting {name_x}-{name_y} on {device}')

    #convert numpy data to tensors (optionally on GPU)
    if train_x is None:
        train_x = torch.tensor(X).float().to(device=device)
    if train_y is None:
        train_y = torch.tensor(Y).float().to(device=device)

    def plot_n_save(model):
        # save weights
        name = '{}_{}'.format(exp_name,get_copula_name_string(model.likelihood.likelihoods))
        weights_filename = '{}/w_{}.pth'.format(path_output,name)
        torch.save(model.gp_model.state_dict(),weights_filename)
        # plot results
        plot_res = '{}/res_{}.png'.format(path_output,name)
        Plot_Fit(model,X,Y,name_x,name_y,plot_res,device=device)

    best_likelihoods = [bvcopula.GaussianCopula_Likelihood()]
    waic_min, model = bvcopula.infer(best_likelihoods,train_x,train_y,device=device)
    # print(get_copula_name_string(best_likelihoods)+f" (WAIC = {waic_min:.4f})")
    plot_n_save(model)

    if waic_min>conf.waic_threshold:
        logging.info("These variables are independent")
        return ([bvcopula.IndependenceCopula_Likelihood()], 0.0)
    else:

        (waic_gumbels, model_gumbels) = bvcopula.infer(conf.gumbel_likelihoods,train_x,train_y,device=device)
        # print(get_copula_name_string(conf.gumbel_likelihoods)+f" (WAIC = {waic_gumbels:.4f})")
        (waic_claytons, model_claytons) = bvcopula.infer(conf.clayton_likelihoods,train_x,train_y,device=device)
        # print(get_copula_name_string(conf.clayton_likelihoods)+f" (WAIC = {waic_claytons:.4f})")

        if waic_min >= min(waic_claytons,waic_gumbels):

            if waic_claytons<waic_gumbels:
                which_leader = important_copulas(model_claytons)
                which_follow = important_copulas(model_gumbels)
                likelihoods_leader = conf.clayton_likelihoods[2:]
                likelihoods_follow = conf.gumbel_likelihoods[2:]
                plot_n_save(model_claytons)
            else:
                which_leader = important_copulas(model_gumbels)
                which_follow = important_copulas(model_claytons)
                likelihoods_leader = conf.gumbel_likelihoods[2:]
                likelihoods_follow = conf.clayton_likelihoods[2:]
                plot_n_save(model_gumbels)

            logging.info(which_leader)
            logging.info(which_follow)
            
            symmetric_part = which_leader[:2] + which_follow[:2] # + = elementwise_or
            assymetric_part = which_leader[2:] + which_follow[2:]

            waic_min = min(waic_claytons,waic_gumbels)
            symmetric_likelihoods = reduce_model(conf.clayton_likelihoods[:2],symmetric_part)
            #print("Symmetric: "+get_copula_name_string(symmetric_likelihoods))
            best_likelihoods = conf.clayton_likelihoods[:2]+likelihoods_leader.copy()
            count_swaps=0
            for iter,i in enumerate(torch.arange(4)[assymetric_part]):
                likelihoods = symmetric_likelihoods.copy()
                if (count_swaps==0) and (iter==torch.sum(assymetric_part).cpu()-1):
                    logging.info('No need to swap the last one, as we already tried that model')
                else:
                    for j in torch.arange(4)[assymetric_part]:
                        if i!=j:
                            likelihoods.append(likelihoods_leader[j])
                        else:
                            likelihoods.append(likelihoods_follow[j])
                    (waic, model) = bvcopula.infer(likelihoods,train_x,train_y,device=device)
                    if waic<waic_min:
                        logging.info("Swap "+get_copula_name_string([likelihoods_leader[i]])+"->"+get_copula_name_string([likelihoods_follow[i]]))
                        # print("Swap "+get_copula_name_string([likelihoods_leader[i]])+"->"+get_copula_name_string([likelihoods_follow[i]]))
                        likelihoods_leader[i] = likelihoods_follow[i]
                        count_swaps+=1
                        waic=waic_min
                        best_likelihoods = likelihoods.copy()
                        which_leader = important_copulas(model)
                        plot_n_save(model)

            #print("Assymetric: "+get_copula_name_string(likelihoods_leader))
       
            if torch.any(which_leader==False):
                best_likelihoods = reduce_model(best_likelihoods,which_leader)
                logging.info("Re-running reduced model...")
                # print("Re-running reduced model...")
                (waic, model) = bvcopula.infer(best_likelihoods,train_x,train_y,device=device)
                # print(get_copula_name_string(best_likelihoods)+f" (WAIC = {waic:.4f})")
                waic_min = waic
                plot_n_save(model)
            else:
                logging.info('Nothing to reduce')
                print("Nothing to reduce")

            # If Frank is still selected, check if Gaussian Copula is better than Frank
            if symmetric_part[1]==True:
                with_gauss = best_likelihoods.copy()
                for i, c in enumerate(with_gauss):
                    if c.name=='Gaussian':
                        with_gauss[i] = bvcopula.FrankCopula_Likelihood()
                #print('Trying Gauss: '+get_copula_name_string(with_gauss))
                (waic, model) = bvcopula.infer(with_gauss,train_x,train_y,device=device)
                if waic<waic_min:
                    logging.info('Frank is better than Gauss')
                    # print('Frank is better than Gauss')
                    waic_min = waic
                    best_likelihoods = with_gauss
                    plot_n_save(model)
            elif len(best_likelihoods)>1:
                # Gaussian is often confused with Clayton+Gumbel or Gumbel+(180-rotated-Gumbel)
                # Check that this did not happen.
                new_best = best_likelihoods #no need to copy here
                for i in range(len(best_likelihoods)-1):
                    for j in range(i,len(best_likelihoods)):
                        if i!=j:
                            logging.info(f"Trying to substitute 2 elements ({i} and {j}) with a Gauss...")
                            # print(f"Trying to substitute 2 elements ({i} and {j}) with a Gauss...")
                            likelihoods = [bvcopula.GaussianCopula_Likelihood()]
                            for k in range(len(best_likelihoods)):
                                if (k!=i) & (k!=j):
                                    likelihoods = likelihoods + [best_likelihoods[k]]
                            (waic, model) = bvcopula.infer(likelihoods,train_x,train_y,device=device)
                            if waic<waic_min:
                                waic_min = waic
                                new_best = likelihoods.copy()
                                # print(get_copula_name_string(new_best)+f" (WAIC = {waic:.4f})")
                                plot_n_save(model)
                best_likelihoods = new_best.copy()
        else: # if Frank was better than all combinations -> Check Gaussian
            waic, model = bvcopula.infer([bvcopula.FrankCopula_Likelihood()],train_x,train_y,device=device)
            if waic<waic_min:
                best_likelihoods = [bvcopula.FrankCopula_Likelihood()]
                waic_min = waic
                # print(get_copula_name_string(best_likelihoods)+f" (WAIC = {waic:.4f})")
                plot_n_save(model)

        # load model
        name = '{}_{}'.format(exp_name,get_copula_name_string(best_likelihoods))
        weights_filename = '{}/w_{}.pth'.format(path_output,name)
        model = bvcopula.load_model(weights_filename, best_likelihoods, device)
        # final reduce
        which = important_copulas(model)
        if torch.any(which==False):
            best_likelihoods = reduce_model(best_likelihoods,which)
            (waic, model) = bvcopula.infer(best_likelihoods,train_x,train_y,device=device)
            if waic>waic_min:
                print('Reducing the model, even though the WAIC gets worse. See logs.')
            waic_min = waic
            print(get_copula_name_string(best_likelihoods)+f" (WAIC = {waic:.4f})")
            plot_n_save(model)
        else:
            logging.info('Nothing to reduce')
    
        print("Final model: "+get_copula_name_string(best_likelihoods))
        logging.info("Final model: "+get_copula_name_string(best_likelihoods))

        name = '{}_{}'.format(exp_name,get_copula_name_string(best_likelihoods))
        source = '{}/w_{}.pth'.format(path_output,name)
        target = '{}/model_{}.pth'.format(path_output,exp_name)
        os.popen('cp {} {}'.format(source,target)) 
        source = '{}/res_{}.png'.format(path_output,name)
        target = '{}/best_{}.png'.format(path_output,exp_name)
        os.popen('cp {} {}'.format(source,target))

        return best_likelihoods, waic_min.cpu().item()
    
