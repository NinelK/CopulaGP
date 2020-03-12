import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import torch
import sys
import time

import bvcopula
import utils

path_models = '/home/nina/models'
path_exp = '/home/nina/VRData/Processing/pkls'
beh = 5

d = {
    'ST260': 104,
    'ST262': 61,
    'ST263': 23,
    'ST264': 34
}

animal = sys.argv[1]#'ST263'
dayN = sys.argv[2]
N_max = d[animal]
day_name = f"Day{dayN}"
exp_pref = f"{animal}_{day_name}"
device = torch.device('cuda:0')

print(exp_pref)

X,_ = utils.load_experimental_data(path_exp, animal, day_name, 0, 1)
S = torch.tensor(X).float().squeeze().to(device)

f_mc_size = 5

def get_MI(n1,n2):
    likelihoods = utils.get_likelihoods(f"{path_models}/{exp_pref}/summary.pkl",n1,n2)
    if (len(likelihoods)==1) & (likelihoods[0].name=='Independence'):
        return (0,0,0,0)
    else:
        weights_file = f"{path_models}/{exp_pref}/model_{n1}-{n2}.pth"

        model = utils.get_model(weights_file, likelihoods, device) 
        if model!=0: #if 0 -- then get_model returned an error, because weights file was not found
            with torch.no_grad():
                Fs = model(S).rsample(torch.Size([f_mc_size])) 
                #[samples_f, copulas, stimuli(positions)]
                MI,sem,Hr,sem1=model.likelihood.stimMI(S,Fs,s_mc_size=200,r_mc_size=20,sR_mc_size=2000,sem_tol=5*1e-3)
                #print(f"{MI.mean().item():.3}±{MI.std().item():.3} -- {(sem.max()/(MI.max()-MI.min())).item():.3}")

            return (MI.mean(),MI.std(),Hr.mean(),Hr.std())
        else:
            return (0,0,0,0)

MI = np.zeros((N_max+beh,N_max+beh,4))

start_time = time.time()
for n1 in range(-5,N_max-1):
    t1 = time.time()
    k=0
    for n2 in range(n1+1,N_max):
        MI[n1+beh,n2+beh,:] = get_MI(n1,n2)
        if MI[n1+beh,n2+beh,0]!=0:
            k+=1
    if k!=0:
        print(n1,' ',(time.time()-t1)/k,' s')
    
total_time = (time.time() - start_time)/60
print(f"Took {total_time:.1f} min overall")

with open(f"{path_models}/MI_measures/{exp_pref}_copulaMI.pkl",'wb') as f:
    pkl.dump(MI,f)
