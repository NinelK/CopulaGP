import torch
from . import conf
import logging
import bvcopula
from utils import get_copula_name_string, Plot_Fit
import os

from .importance import important_copulas, reduce_model
   
def select_light(X: torch.Tensor, Y: torch.Tensor, device: torch.device,
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

    def checkNreduce(waic,model,likelihoods,
        scnd_best_waic,scnd_best_model_data,scnd_best_lik):
        which = important_copulas(model)
        if torch.any(which==False):
            likelihoods_new = reduce_model(likelihoods,which)
            if get_copula_name_string(likelihoods_new)!=get_copula_name_string(scnd_best_lik):
                logging.info("Re-running reduced model...")
                (waic_new, model_new) = bvcopula.infer(likelihoods_new,train_x,train_y,device=device)
                logging.info(get_copula_name_string(likelihoods_new)+f" (WAIC = {waic:.4f})")
                return (waic_new,likelihoods_new,model_new.serialize())
            else:
                logging.info("Reduced to the previous model.")
                return (scnd_best_waic,scnd_best_lik,scnd_best_model_data)
        else:
            logging.info('Nothing to reduce')
            return (waic,likelihoods,model.serialize())

    best_likelihoods = [bvcopula.GaussianCopula_Likelihood()]
    waic_min, model = bvcopula.infer(best_likelihoods,train_x,train_y,device=device)
    best_model = model.serialize()
    logging.info(get_copula_name_string(best_likelihoods)+f" (WAIC = {waic_min:.4f})")
    
    if waic_min>conf.waic_threshold:
        logging.info("These variables are independent")
        best_likelihoods = [bvcopula.IndependenceCopula_Likelihood()]
        best_model = bvcopula.Pair_CopulaGP(best_likelihoods).serialize()
    else:
        (waic_claytons, model_claytons) = bvcopula.infer(conf.clayton_likelihoods,train_x,train_y,device=device)
        logging.info(get_copula_name_string(conf.clayton_likelihoods)+f" (WAIC = {waic_claytons:.4f})")

        if waic_min >= waic_claytons:
            
            waic_min, best_likelihoods, best_model = checkNreduce(waic_claytons,model_claytons,conf.clayton_likelihoods,
                                               waic_min,best_model,[bvcopula.GaussianCopula_Likelihood()])
            #try adding Frank
            with_frank = [bvcopula.FrankCopula_Likelihood()] + best_likelihoods
            (waic, model) = bvcopula.infer(with_frank,train_x,train_y,device=device)
            if waic<waic_min:
                logging.info('Frank added')
                waic_min, best_likelihoods, best_model = checkNreduce(waic,model,with_frank,
                                                waic_min,best_model,best_likelihoods)
            else:
                logging.info('Frank is not helping')
                
        else: # if Gaussian was better than all combinations -> Check Frank
            waic, model = bvcopula.infer([bvcopula.FrankCopula_Likelihood()],train_x,train_y,device=device)
            if waic<waic_min:
                best_likelihoods = [bvcopula.FrankCopula_Likelihood()]
                waic_min = waic
                best_model = model.serialize()
                print(get_copula_name_string(best_likelihoods)+f" (WAIC = {waic:.4f})")

        logging.info("Final model: "+get_copula_name_string(best_likelihoods))

    return best_model, waic_min
    
