# Assuming directory structure:
#    ./dataset/vcc2016/wav/Training Set/SF1/100001.wav 

import os
from os.path import join
from scipy.interpolate import interp1d
from scipy.signal import firwin
from scipy.signal import lfilter
import librosa
import numpy as np
import pyworld as pw
import tensorflow as tf
from sklearn import preprocessing
import pycwt as wavelet


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir_to_wav', '../data/wav', 'Dir to *.wav')
tf.app.flags .DEFINE_string('dir_to_bin', './data_multi/bin', 'Dir to output *.bin')
tf.app.flags.DEFINE_integer('fs', 16000, 'Global sampling frequency')
tf.app.flags.DEFINE_float('f0_ceil', 500, 'Global f0 ceiling')

FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
CWT_DIM = SP_DIM
FEAT_DIM = 1 + CWT_DIM * 2 + 1 + 1   # [lf0_cwt, f0, lf0_cwt_norm, uv_f0, s]1029
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`

EPSILON = 1e-10


def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    ''' Extract WORLD feature from waveform '''
    _f0, t = pw.dio(x, fs, f0_ceil=args.f0_ceil)            # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size) # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }


def list_dir(path):
    ''' retrieve the 'short name' of the dirs '''
    return sorted([f for f in os.listdir(path) if os.path.isdir(join(path, f))])


def list_full_filenames(path):
    ''' return a generator of full filenames '''
    return (
        join(path, f)
            for f in os.listdir(path)
                if not os.path.isdir(join(path, f)))
# cwt
def low_pass_filter(x, fs, cutoff=70, padding=True):
    """FUNCTION TO APPLY LOW PASS FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), 'edge')
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2: -numtaps // 2]
    return lpf_x

def convert_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float64(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        logging.warn("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0

def get_cont_lf0(f0, frame_period=5.0):
    uv, cont_f0_lpf = convert_continuos_f0(f0)
    cont_f0_lpf = low_pass_filter(cont_f0_lpf, int(1.0 / (frame_period * 0.001)), cutoff=20)
    cont_lf0_lpf = np.log(cont_f0_lpf)
    return uv, cont_lf0_lpf

def get_lf0_cwt(lf0):
    mother = wavelet.MexicanHat()
    dt = 0.005
    dj = 0.015
    s0 = dt*2
    J = 513 - 1
    Wavelet_lf0, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(np.squeeze(lf0), dt, dj, s0, J, mother)
    Wavelet_lf0 = np.real(Wavelet_lf0).T
    return Wavelet_lf0, scales

def inverse_cwt(Wavelet_lf0,scales):
    lf0_rec = np.zeros([Wavelet_lf0.shape[0],len(scales)])
    for i in range(0,len(scales)):
        lf0_rec[:,i] = Wavelet_lf0[:,i]*((i+200+2.5)**(-2.5))
    lf0_rec_sum = np.sum(lf0_rec,axis = 1)
    lf0_rec_sum_norm = preprocessing.scale(lf0_rec_sum)
    return lf0_rec_sum_norm

def norm_scale(Wavelet_lf0):
    Wavelet_lf0_norm = np.zeros((Wavelet_lf0.shape[0], Wavelet_lf0.shape[1]))
    mean = np.zeros((1,Wavelet_lf0.shape[1]))#[1,10]
    std = np.zeros((1, Wavelet_lf0.shape[1]))
    for scale in range(Wavelet_lf0.shape[1]):
        mean[:,scale] = Wavelet_lf0[:,scale].mean()
        std[:,scale] = Wavelet_lf0[:,scale].std()
        Wavelet_lf0_norm[:,scale] = (Wavelet_lf0[:,scale]-mean[:,scale])/std[:,scale]
    return Wavelet_lf0_norm, mean, std

def extract(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction ''' 
    x, _ = librosa.load(filename, sr=args.fs, mono=True, dtype=np.float64)
    features = wav2pw(x, args.fs, fft_size=fft_size)
    f0 = features['f0']
    uv_f0, lf0_cont = get_cont_lf0(f0)
    uv_f0 = uv_f0.reshape([-1,1])
    lf0_cont = (lf0_cont - lf0_cont.mean())/lf0_cont.std()
    lf0_cwt, scales = get_lf0_cwt(lf0_cont)
    lf0_cwt_norm, _, _ = norm_scale(lf0_cwt)

    f0 = f0.reshape([-1,1])
    np.save('scales.npy',scales)
    return np.concatenate([lf0_cwt, f0, lf0_cwt_norm, uv_f0], axis=1).astype(dtype)


def extract_and_save_bin_to(dir_to_wav, dir_to_bin, speakers):
    '''
    NOTE: the directory structure must be [args.dir_to_wav]/[Set]/[speakers]
    '''
    counter = 1
    N = len(tf.gfile.Glob(join(dir_to_wav, '*', '*', '*.wav')))
    for d in list_dir(dir_to_wav):  # ['Training Set', 'Testing Set']
        path = join(dir_to_wav, d)
        for s in list_dir(path):  # ['SF1', ..., 'TM3']
            path = join(dir_to_wav, d, s)
            output_dir = join(dir_to_bin, d, s)
            tf.gfile.MakeDirs(output_dir)
            for f in list_full_filenames(path):  # ['10001.wav', ...]
                print('\rFile {}/{}: {:50}'.format(counter, N, f), end='')
                features = extract(f)
                labels = speakers.index(s) * np.ones(
                    [features.shape[0], 1],
                    np.float32,
                )
                b = os.path.splitext(f)[0]
                _, b = os.path.split(b)
                features = np.concatenate([features, labels], 1)
                with open(join(output_dir, '{}.bin'.format(b)), 'wb') as fp:
                    fp.write(features.tostring())
                counter += 1
        print()


def read_f0(
    file_pattern,
    batch_size,
    record_bytes=RECORD_BYTES,
    capacity=2048,
    min_after_dequeue=1536,
    num_threads=8,
    data_format='NCHW',
    normalizer=None,
    ):
    '''
    Read only `sp` and `speaker`
    Return:
        `feature`: [b, c]
        `speaker`: [b,]
    '''
    with tf.device('cpu'):
        with tf.name_scope('InputSpectralFrame'):
            files = tf.gfile.Glob(file_pattern)
            filename_queue = tf.train.string_input_producer(files)


            reader = tf.FixedLengthRecordReader(record_bytes)
            _, value = reader.read(filename_queue)
            value = tf.decode_raw(value, tf.float32)

            value = tf.reshape(value, [FEAT_DIM,])
            #feature = value[:SP_DIM]   # NCHW format
            feature = value[513+1:513*2+1]

            if normalizer is not None:
                feature = normalizer.forward_process(feature)

            if data_format == 'NCHW':
                feature = tf.reshape(feature, [1, 513, 1])
            elif data_format == 'NHWC':
                feature = tf.reshape(feature, [513, 1, 1])
            else:
                pass

            speaker = tf.cast(value[-1], tf.int64)
            #zk
           # f0 = tf.cast(value[SP_DIM * 2], tf.int64)

            return tf.train.shuffle_batch(
                [feature, speaker],
                batch_size,
                capacity=capacity,
                min_after_dequeue=min_after_dequeue,
                num_threads=num_threads,
                # enqueue_many=True,
            )

def read_whole_features(file_pattern, num_epochs=1):
    '''
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    '''
    with tf.device('cpu'):
        with tf.name_scope('InputPipline'):
            files = tf.gfile.Glob(file_pattern)
            print('{} files found'.format(len(files)))
            filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            print("Processing {}".format(key), flush=True)
            value = tf.decode_raw(value, tf.float32)
            value = tf.reshape(value, [-1, FEAT_DIM])
            return {
                'lf0_cwt': value[:, :513],
                'f0': value[:, 513],
                'lf0_cwt_norm': value[:, 513+1:513*2+1],
                'uv': value[:,513*2+1],
                'speaker': tf.cast(value[:, -1], tf.int64),
                'filename': key,
            }

def update_whole_features(features, file_pattern, num_epochs=1):
    '''
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    '''
    with tf.device('cpu'):
        with tf.name_scope('InputPipline'):
            files = tf.gfile.Glob(file_pattern)
            print('{} files found'.format(len(files)))
            filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
            reader = tf.WholeFileReader()
            key, value = reader.read(filename_queue)
            print("Processing {}".format(key), flush=True)
            value = tf.decode_raw(value, tf.float32)
            value = tf.reshape(value, [-1, FEAT_DIM])
            return {
                'lf0_cwt': value[:, :513],
                'f0': value[:, 513],
                'lf0_cwt_norm': value[:, 513+1:513*2+1],
                'uv': value[:,513*2+1],
                'speaker': tf.cast(value[:, -1], tf.int64),
                'filename': key,
            }
def dic2npy(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''

    lf0_cwt = features['lf0_cwt']
    uv = np.reshape(features['uv'], [-1, 1])
    speaker = np.reshape(features['speaker'],[-1,1])
    filename = features['filename']
    feats = np.concatenate([lf0_cwt,uv,speaker],axis=1)
    feats = np.float32(feats)
    return feats


def make_speaker_tsv(path):
    speakers = []
    for d in list_dir(path):
        speakers += list_dir(join(path, d))
    speakers = sorted(set(speakers))
    with open('./etc/speakers.tsv', 'w') as fp:
        for s in speakers:
            fp.write('{}\n'.format(s))
    return speakers

if __name__ == '__main__':
    speakers = make_speaker_tsv(args.dir_to_wav)
    extract_and_save_bin_to(
        args.dir_to_wav,
        args.dir_to_bin,
        speakers,
    )
