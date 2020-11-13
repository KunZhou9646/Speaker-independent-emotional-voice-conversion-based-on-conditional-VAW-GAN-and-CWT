
import os
import numpy as np
import pyworld as pw
import soundfile as sf
from sklearn import preprocessing
from os.path import join

def inverse_cwt(Wavelet_lf0,scales):
    lf0_rec = np.zeros([Wavelet_lf0.shape[0],len(scales)])
    for i in range(0,len(scales)):
        lf0_rec[:,i] = Wavelet_lf0[:,i]*((i+200+2.5)**(-2.5))
    lf0_rec_sum = np.sum(lf0_rec,axis = 1)
    lf0_rec_sum = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum

def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = np.zeros((1,Wavelet_lf0.shape[1]))#[1,10]
    std = np.zeros((1, Wavelet_lf0.shape[1]))
    for scale in range(Wavelet_lf0.shape[1]):
        mean[:,scale] = Wavelet_lf0[:,scale].mean()
        std[:,scale] = Wavelet_lf0[:,scale].std()
        Wavelet_lf0_norm[:,scale] = (Wavelet_lf0[:,scale]-mean[:,scale])/std[:,scale]
    return Wavelet_lf0_norm, mean, std

def denormalize(Wavelet_lf0_norm, mean, std):
    Wavelet_lf0_denorm = np.zeros((Wavelet_lf0_norm.shape[0], Wavelet_lf0_norm.shape[1]))
    for scale in range(Wavelet_lf0_norm.shape[1]):
        Wavelet_lf0_denorm[:,scale] = Wavelet_lf0_norm[:,scale]*std[:,scale]+mean[:,scale]
    return Wavelet_lf0_denorm

def post_processing(features, scales, mean_f0_target, std_f0_target, para_lf0_cwt):



    lf0_cwt = features[:,:513]
    uv = features[:,513]

    lf0_cwt_norm,_,_ = norm_scale(lf0_cwt)
    mean_lf0_cwt = para_lf0_cwt[:513]
    std_lf0_cwt = para_lf0_cwt[513:]
    lf0_cwt = lf0_cwt * std_lf0_cwt + mean_lf0_cwt

    f0 = inverse_cwt(lf0_cwt, scales)

    f0_converted = f0 * std_f0_target + mean_f0_target

    f0_t = np.squeeze(uv) * np.exp(f0_converted)
    f0_t = np.ascontiguousarray(f0_t)
    f0_t = np.float64(f0_t)

    return f0_t

source_data_dir = './f0_results'
if not os.path.isdir('./final_results'):

   os.mkdir('./final_results')


freq = 16000
frame_period = 5.0
SP_DIM = 513
CWT_DIM = 513

scales = np.load('./scales.npy')
target_f0_parameter = np.fromfile('./etc/Angryf0.npf')
para_lf0_cwt = np.fromfile('./etc/Angry_lf0_cwt.npf')
mean_f0_target = target_f0_parameter[0]
std_f0_target  = target_f0_parameter[1]




for file in os.listdir(source_data_dir):
    basename = os.path.splitext(file)[0]
    filepath = os.path.join(source_data_dir, file)
    features = np.fromfile(filepath,np.float32)
    features = features.reshape([-1,515])
    f0_t= post_processing(features, scales, mean_f0_target, std_f0_target, para_lf0_cwt)
    with open(join('./final_results', '{}.npy'.format(basename)), 'wb') as fp:
        np.save(fp,f0_t)
