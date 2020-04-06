import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import bvcopula
import utils

import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
from scipy.stats import pearsonr
from scipy.stats import kendalltau

animal = 'ST260'
dayN = 3
day_name = 'Day{}'.format(dayN)
path = '/home/nina/VRData/Processing/pkls'

out = 'figures_cosyne_new'

mrg = 0.2

device=int(sys.argv[1])
n1 = int(sys.argv[2])
n2 = int(sys.argv[3])
name1, name2 = f"Neuron {n1}", f"Neuron {n2}"
print(name1,name2)
X,Y = utils.load_experimental_data(path, animal, day_name, n1, n2)

NSamp = X.shape[0]

X_pos = X.squeeze()*160
Y_pos = Y
ds = 2.0
x = np.linspace(0,1,int(160/ds))
p = np.empty(int(160/ds))
k = np.empty(int(160/ds))
for b in range(int(160/ds)):
    dat = Y_pos[(X_pos>b*ds) & (X_pos<(b+1)*ds)]
    if len(dat)>1:
        p[b] = pearsonr(*dat.T)[0]
        k[b] = kendalltau(*dat.T)[0]
p = np.convolve(np.array(p), np.ones((4,))/4, mode='valid')    
x=x[2:-1]

filename = f"../models/{animal}_{day_name}/summary.pkl"
with open(filename,'rb') as f:
    summary = pkl.load(f)
    
print(summary[n1+5,n2+5][1])
likelihoods = summary[n1+5,n2+5][0]
#likelihoods = [bvcopula.FrankCopula_Likelihood(), bvcopula.GumbelCopula_Likelihood(rotation='0°')]
#print(likelihoods)

#convert numpy data to tensors (optionally on GPU)
train_x = torch.tensor(X).float().cuda(device=device)
train_y = torch.tensor(Y).float().cuda(device=device)

waic, model = bvcopula.infer(likelihoods,train_x,train_y,device=device)

# define test set (optionally on GPU)
denser = 2 # make test set 2 times denser then the training set
testX = np.linspace(0,1,100)
test_x = torch.tensor(testX).float().cuda(device=device)

# visualize the result
fig, ax = plt.subplots(1,2,figsize=(12, 2))
    
for a in ax:
    a.axvline(120, color='black', alpha=0.5)
    a.axvline(140, color='black', alpha=0.5)
    a.axvline(160, color='black', alpha=0.5)    
    
utils.Plot_MixModel_Param_MCMC(ax,model,test_x,testX*160,rho=np.array([160*x,p]),title='for {} vs {}'.format(name1,name2))

plt.subplots_adjust(hspace=0.7)

fig.savefig(f"{out}/parameters_{n1}_{n2}.pdf")

s_mc_size=200
S = torch.linspace(0,1,s_mc_size).to(device)
f_mc_size = 20
with torch.no_grad():
    Fs = model(S).rsample(torch.Size([f_mc_size])) 
    #[samples_f, copulas, stimuli(positions)]
    copula = model.likelihood(Fs) #[f, stimuli(positions)]
    H = copula.entropy()
MI_nl = -H.mean(dim=0).cpu().numpy()
dMI_nl = -H.std(dim=0).cpu().numpy()

waic, model_l = bvcopula.infer([bvcopula.FrankCopula_Likelihood()],train_x,train_y,device=device)
with torch.no_grad():
    Fs = model_l(S).rsample(torch.Size([f_mc_size])) 
    #[samples_f, copulas, stimuli(positions)]
    copula = model_l.likelihood(Fs) #[f, stimuli(positions)]
    H = copula.entropy()
MI_l = -H.mean(dim=0).cpu().numpy()
dMI_l = -H.std(dim=0).cpu().numpy()

fig = plt.figure(figsize=(5, 2))
plt.axvline(120, color='black', alpha=0.5)
plt.axvline(140, color='black', alpha=0.5)
plt.axvline(160, color='black', alpha=0.5)
#mid = (ints[1:]+ints[:-1])/2*160
# n1,n2 = 3,63
# name1,name2 = f"{n1}",f"{n2}"
mid = 160*S.cpu()
plt.plot(mid,MI_l)
plt.fill_between(mid,(MI_l-dMI_l),(MI_l+dMI_l),alpha=0.3,label='Frank copula')#,label="MI={:.2}±{:.1}".format(MI,dMI))
plt.plot(mid,MI_nl)
plt.fill_between(mid,(MI_nl-dMI_nl),(MI_nl+dMI_nl),alpha=0.3,label='Best of Copula-GP')#,label="MI={:.2}±{:.1}".format(MI,dMI))
plt.title('MI between {} and {}'.format(name1,name2))
plt.legend(loc=2)
plt.xlabel('Position, [cm]')
plt.ylabel('MI, bits')
plt.xlim(0,160)
fig.savefig(f"{out}/MI_{n1}_{n2}.pdf")

# define test set (optionally on GPU)
testX = np.repeat(X,5)
test_x = torch.tensor(testX).float().cuda(device=device)
#generate some samples
model.eval()
with gpytorch.settings.num_likelihood_samples(1):
    copula = model.likelihood(model(test_x).mean)
    test_y = copula.rsample()
    Y_sim = test_y.cpu().detach().numpy()

# visualize the result
fig = plt.figure(figsize=(5, 2))
bottom_axes = np.array([fig.add_axes([0,0,0.2,0.5]),
               fig.add_axes([0.25,0,0.2,0.5]),
               fig.add_axes([0.5,0,0.2,0.5]),
               fig.add_axes([0.75,0,0.2,0.5])])

plt.subplots_adjust(hspace=0.5)

bottom_axes[0].set_ylabel(name2)
bottom_axes[0].set_xlabel(name1)

interval_ends = [0,60,120,140,160]
utils.Plot_Copula_Density(bottom_axes, testX.squeeze()*160, Y_sim.squeeze(), interval_ends, shade=True)
utils.Plot_Copula_Density(bottom_axes, X.squeeze()*160, Y, interval_ends, shade=False, color='#073763ff')

plt.subplots_adjust(wspace=0.5)
fig.savefig(f"{out}/densities_{n1}_{n2}.pdf")
