import marginal as mg
import pickle as pkl
import numpy as np
import time
import sys

animal = sys.argv[1]#'ST263'
dayN = sys.argv[2]
day_name = f"Day{dayN}"
exp_pref = f"{animal}_{day_name}"

print(exp_pref)

path = '/home/nina/VRData/Processing/pkls'
path_models = '/home/nina/models'

with open("{}/{}_{}_signals.pkl".format(path,animal,day_name),'rb') as f:
    signal_pkl = pkl.load(f)
with open("{}/{}_{}_behaviour.pkl".format(path,animal,day_name),'rb') as f:
    behaviour_pkl = pkl.load(f)
for s in ['ROIsN','trialStart','maxTrialNum','trials']:
    assert(np.allclose(signal_pkl[s],behaviour_pkl[s]))
stimulus = (behaviour_pkl['position']/160)%1

repeats = 10
start_time = time.time()
gao_MIs = np.zeros((signal_pkl['signals_fissa'].shape[0]+5,repeats))
for rep in range(repeats):
    order = np.random.choice(len(stimulus), int(len(stimulus)/2), replace=False)
    gao_MIs[0,rep] = mg.revised_mi(stimulus[order].reshape(-1,1).tolist(),\
                                   behaviour_pkl['transformed_late_reward'][order].reshape(-1,1).tolist())
    gao_MIs[1,rep] = mg.revised_mi(stimulus[order].reshape(-1,1).tolist(),\
                                   behaviour_pkl['transformed_early_reward'][order].reshape(-1,1).tolist())
    gao_MIs[2,rep] = mg.revised_mi(stimulus[order].reshape(-1,1).tolist(),\
                                  (behaviour_pkl['transformed_late_reward']+\
                                   behaviour_pkl['transformed_early_reward'])[order].reshape(-1,1).tolist())
    gao_MIs[3,rep] = mg.revised_mi(stimulus[order].reshape(-1,1).tolist(),\
                                   behaviour_pkl['transformed_licks'][order].reshape(-1,1).tolist())
    gao_MIs[4,rep] = mg.revised_mi(stimulus[order].reshape(-1,1).tolist(),\
                                   behaviour_pkl['transformed_velocity'][order].reshape(-1,1).tolist())
    for i, r in enumerate(signal_pkl['signals_fissa']):
        gao_MIs[i+5,rep] = mg.revised_mi(stimulus[order].reshape(-1,1).tolist(),r[order].reshape(-1,1).tolist())
        
all_time = (time.time()-start_time)/60
print(f"Took {all_time:.1f} min")

spike_H = np.zeros(signal_pkl['spikes'].shape[0]+5)
def get_entropy(x):
    states = np.unique(x)
    H = 0
    for s in states:
        p = np.sum(x==s)/len(x)
        H -= p*np.log(p)
    return H/np.log(len(states)) #divide by the maximal possible entropy
for i, x in enumerate(signal_pkl['spikes']):
    spike_H[i+5] = get_entropy(x)
spike_H[0] = get_entropy(behaviour_pkl['discrete_LR'])
spike_H[1] = get_entropy(behaviour_pkl['discrete_ER'])
spike_H[2] = get_entropy(behaviour_pkl['discrete_LR']+behaviour_pkl['discrete_ER'])
spike_H[3] = get_entropy(behaviour_pkl['discrete_licks'])
spike_H[4] = get_entropy(behaviour_pkl['discrete_vel'])
print('spikes done')

# means = gao_MIs.mean(axis=-2)
# stds = gao_MIs.std(axis=-2)
# result = np.stack([means[:,0],stds[:,0],means[:,1],stds[:,1]])

result = np.stack([gao_MIs.mean(axis=-1),gao_MIs.std(axis=-1),spike_H])

with open(f"{path_models}/MI_measures/{exp_pref}_marginalMI.pkl",'wb') as f:
    pkl.dump(result,f)
