import numpy as np
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/nina/LFI/')
import bvcopula

path = '../out_christmas/'
animal = sys.argv[1]
dayN = sys.argv[2]
day_name = 'Day{}'.format(dayN)
exp_pref = '{}_{}'.format(animal,day_name)
beh = 5 #number of behavioural variables
list_files = glob.glob(path+exp_pref+"*.pkl")

with open(list_files[0],'rb') as f:
     new_data = pkl.load(f)
data_pkls = np.empty_like(new_data) #create an empty array

#parse text results

def neuron_ids(string):
    arr = str.split(string,sep='-')
    n1,n2 = 1,1
    which = 0
    for a in arr:
        try:
            n = int(a)
            if which:
                n2*=n
            else:
                n1*=n
                which=1
        except ValueError:
            if which:
                n2 = -1
            else:
                n1 = -1
    return n1,n2

import csv

beh = 5
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

###############################################
# take action here
# data_txt[0,2] = ['Independence','0','0',-5,-3]  
# print(data_txt[5,6])
# data_txt[0,5] = data_txt[5,6]
# data_txt[33+5,40+5] = data_txt[5,6]
##############################################

exit()

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

#repair lost pkls
from select_copula import elements
from utils import get_copula_name_string

def create_copula_dict(elements):
    d = {}
    for el1 in elements:
        d[get_copula_name_string([el1])] = [el1]
        for el2 in elements:
            d[get_copula_name_string([el1,el2])] = [el1,el2]
            for el3 in elements:
                d[get_copula_name_string([el1,el2,el3])] = [el1,el2,el3]
                for el4 in elements:
                    d[get_copula_name_string([el1,el2,el3,el4])] = [el1,el2,el3,el4]
    return d

copula_dict = create_copula_dict(elements)

for n1 in range(0,NN-1+beh):
    for n2 in range(n1+1,NN+beh):
        if (data_pkls[n1,n2]==None) & (data_txt[n1,n2]!=None):
            data_pkls[n1,n2] = [copula_dict[data_txt[n1,n2][0]],data_txt[n1,n2][0],
                                float(data_txt[n1,n2][1]), int(data_txt[n1,n2][2])]
        elif (data_pkls[n1,n2]!=None):
            if data_pkls[n1,n2][1]!=data_txt[n1,n2][0]:
                 print('What? {}-{}'.format(n1,n2))


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
