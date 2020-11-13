import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

if not os.path.isdir('./test_condition_results/Neutral'):

   os.mkdir('./test_condition_results/Neutral')

file_all_dir = '../data/bin/evaluation_set/Neutral'

file_f0_dir = '../vaw-gan-f0/final_results'

file_results_dir = './test_condition_results/Neutral'

for file in os.listdir(file_all_dir):
    basename = os.path.splitext(file)[0]
    filepath = os.path.join(file_all_dir, file)
    features = np.fromfile(filepath,np.float32)
    features = features.reshape([-1,1029])
    sp = features[:,:513]
    ap = features[:,513:513*2]
    f0 = features[:,-3].reshape([-1,1])
    en = features[:,-2].reshape([-1,1])
    s = features[:,-1].reshape([-1,1])
    
    f0_t = np.load(os.path.join(file_f0_dir,basename + '.npy'))
    f0_t = np.float32(f0_t).reshape([-1,1])
    features_new = np.concatenate([sp,ap,f0_t,en,s],axis=1)
    
    with open(join(file_results_dir,'{}.bin'.format(basename)),'wb') as fp:
        fp.write(features_new.tostring())
