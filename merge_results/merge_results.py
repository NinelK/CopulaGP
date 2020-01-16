import numpy as np
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/nina/LFI/')

path = '../out_christmas/'
animal = sys.argv[1]
dayN = sys.argv[2]
day_name = 'Day{}'.format(dayN)
exp_pref = '{}_{}'.format(animal,day_name)
beh = 5 #number of behavioural variables
list_files = glob.glob(path+exp_pref+"*.pkl")

# print(list_files)

# First merge pkls

def fix_zero_times(entry):
    # this function deletes stopped or broken runs
    if (entry!=None):
        if (entry[-1]==0) & (entry[1]!='Independence'):
            return None
        else:
            return entry
    else:
        return None

with open(list_files[0],'rb') as f:
    new_data = pkl.load(f)
data_pkls = np.empty_like(new_data) #create an empty array
for file in list_files:
    with open(file,'rb') as f:
        print(file)
        new_data = pkl.load(f)
        new_data = np.array([[fix_zero_times(j) for j in i] for i in new_data])
        # it is important to fix the data(above) before adding to the results
        #check rewrites
        rewrite_rule = (new_data!=None) & (data_pkls!=None)
        rewrite_new = new_data[rewrite_rule]
        if len(rewrite_new)>0:
            overwrite = data_pkls[rewrite_rule]
            X,Y = np.mgrid[0:new_data.shape[0],0:new_data.shape[1]]
            for old_el,new_el,x,y in zip(overwrite,rewrite_new,X[rewrite_rule],Y[rewrite_rule]):
                print('Overwrite {}({}) with {}({}) at {}-{}'.format(
                    old_el[1],old_el[2],
                    new_el[1],new_el[2],
                    x-beh,y-beh
                    ))
                print(new_el[-1])
        #add results
        data_pkls[new_data!=None] = new_data[new_data!=None]

print(f'{path}{exp_pref}_pkls_content.png')
plt.imsave(f'{path}{exp_pref}_pkls_content.png',((data_pkls!=None)*1.0))
