import numpy as np
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/nina/CopulaGP/')

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

#exit()

NN = data_pkls.shape[0]-beh
data_txt = np.empty((NN+beh,NN+beh),dtype=object)

with open(path+animal+'_'+day_name+'_model_list.txt','r') as f:
    next(f) # skip headings
    reader=csv.reader(f,delimiter='\t')
    for res,waic,time in reader:
        neuron, copula = str.split(res)
        n1,n2 = neuron_ids(neuron)
        assert(n1<NN)
        assert(n2<NN)
        if time!=0:
            data_txt[n1+beh,n2+beh] = [copula,waic,time,n1,n2]

plt.imsave(f'{path}{exp_pref}_txt_content.png',((data_txt!=None)*1.0))

# here we check if there are any missing elements
check_nones = 0
for i in range(-beh,NN-1):
    for j in range(i+1,NN):
        if data_txt[i+beh,j+beh]==None:
            check_nones = 1
            print(f"Empty: {i}-{j}")

if check_nones:
    print('Fix all missing elements before we can proceed!!!')
    exit()
else:
    print('All data are present!')

# now check that everything is merged correctly and save

def check_data_merge(data_pkls, data_txt):
    err = 0
    for pkls,txt in zip(data_pkls.flatten(), data_txt.flatten()):
        if pkls!=None:
            assert txt!=None
            if pkls[1]!=txt[0]:
                err = 1
        #    print(pkl[1],txt[0],pkl[2],txt[1],txt[-2],txt[-1])
    return err

def try_copy(source,target):
    try:
        os.path.exists(source)
    except FileExistsError as error:
        print(error)
        return 0
    finally:
        os.popen('cp {} {}'.format(source,target)) 
        return 1
        
def copy_all_weights(data,in_dir,out_dir):
    for i in range(data.shape[0]):
        for j in range(i+1,data.shape[1]):
            if data[i,j]!=None:
                if data[i,j][1]!='Independence':
                    name = '{}-{}'.format(i-beh,j-beh)
                    source = '{}/model*_{}.pth'.format(in_dir,name)
                    target = '{}/model_{}.pth'.format(out_dir,name)
                    assert try_copy(source,target)
                    source = '{}/best*_{}.png'.format(in_dir,name)
                    target = '{}/figs/best_{}.png'.format(out_dir,name)
                    assert try_copy(source,target)
    with open(out_dir+'/summary.pkl','wb') as f:
        pkl.dump(data,f)

import os

upper_ones = np.triu(np.ones(data_pkls.shape[0]))-np.diag(np.ones(data_pkls.shape[0]))

out_dir = '../../models/'+exp_pref
try:
    os.mkdir(out_dir)
    os.mkdir(out_dir+'/figs/')
except FileExistsError as error:
    list_files = glob.glob(out_dir+"*.p*")
    if len(list_files):
        raise Exception("Folder not empty, save nothing")

assert check_data_merge(data_pkls, data_txt)==0
assert np.all(1*(data_pkls!=None) == upper_ones) #check that all results exist
# if folder is empty and all merged well -- save
copy_all_weights(data_pkls,path+exp_pref,out_dir)
