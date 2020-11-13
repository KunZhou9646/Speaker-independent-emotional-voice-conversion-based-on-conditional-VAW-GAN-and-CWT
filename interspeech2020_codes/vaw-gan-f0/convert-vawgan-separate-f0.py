import json
import os
import sys
from os.path import join
import tensorflow as tf
import numpy as np
import soundfile as sf

from util.wrapper import load
from analyzer_f0 import read_whole_features,dic2npy
from datetime import datetime
from importlib import import_module

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('corpus_name', 'emotion_vc', 'Corpus name')

#tf.app.flags.DEFINE_string('checkpoint_f0_cwt', './logdir/train/0314-1323-50-2020/model.ckpt-46860', 'root of log dir')
tf.app.flags.DEFINE_string('checkpoint_f0_cwt', './logdir/train/[TIMESTEP]/model.ckpt-[MODEL ID]', 'root of log dir')

tf.app.flags.DEFINE_string('src', 'Neutral', 'source speaker')
tf.app.flags.DEFINE_string('trg', 'Angry', 'target speaker')
tf.app.flags.DEFINE_string('output_dir', './logdir', 'root of output dir')

tf.app.flags.DEFINE_string('module', 'model.vawgan', 'Module')
tf.app.flags.DEFINE_string('module_original', 'model.vawgan_original', 'Module')
tf.app.flags.DEFINE_string('module_original_en', 'model.vawgan_original_en', 'Module')

tf.app.flags.DEFINE_string('model', 'VAWGAN', 'Model')
tf.app.flags.DEFINE_string('file_pattern', './data_multi/bin/evaluation_set/{}/*.bin', 'file pattern')
tf.app.flags.DEFINE_string(
    'speaker_list', './etc/speakers.tsv', 'Speaker list (one speaker per line)'
)



def make_output_wav_name(output_dir, filename):
    basename = str(filename, 'utf8')
    basename = os.path.split(basename)[-1]
    basename = os.path.splitext(basename)[0]
    return os.path.join(
        output_dir, 
        '{}-{}-{}.bin'.format(args.src, args.trg, basename)
    )

def make_output_bin_name(output_dir, filename):
    basename = str(filename, 'utf8')
    basename = os.path.split(basename)[-1]
    basename = os.path.splitext(basename)[0]
    return basename


def get_default_output(logdir_root):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, 'output', STARTED_DATESTRING)
    print('Using default logdir: {}'.format(logdir))        
    return logdir

def convert_f0(f0, src, trg):
    mu_s, std_s = np.fromfile(os.path.join('./etc', '{}.npf'.format(src)), np.float32)
    mu_t, std_t = np.fromfile(os.path.join('./etc', '{}.npf'.format(trg)), np.float32)
    lf0 = tf.where(f0 > 1., tf.log(f0), f0)
    lf0 = tf.where(lf0 > 1., lf0 * std_t + mu_t, lf0)
    lf0 = tf.where(lf0 > 1., tf.exp(lf0), lf0)
    return lf0


def nh_to_nchw(x):
    with tf.name_scope('NH_to_NCHW'):
        x = tf.expand_dims(x, 1)      # [b, h] => [b, c=1, h]
        return tf.expand_dims(x, -1)  # => [b, c=1, h, w=1]

def nh_to_nhwc(x):
    with tf.name_scope('NH_to_NHWC'):
        return tf.expand_dims(tf.expand_dims(x, -1), -1)


def main(unused_args=None):

    if args.model is None:
        raise ValueError(
            '\n  You MUST specify `model`.' +\
            '\n    Use `python convert.py --help` to see applicable options.'
        )


    module = import_module(args.module_original, package=None)
    MODEL = getattr(module, args.model)

    FS = 16000

    with open(args.speaker_list) as fp:
        SPEAKERS = [l.strip() for l in fp.readlines()]


    logdir_f0, ckpt_f0_cwt = os.path.split(args.checkpoint_f0_cwt)


# f0:
    if 'VAE' in logdir_f0:
        _path_to_arch, _ = os.path.split(logdir_f0)
    else:
        _path_to_arch = logdir_f0
    arch_f0 = tf.gfile.Glob(os.path.join(_path_to_arch, 'architecture*.json'))
    if len(arch_f0) != 1:
        print('WARNING: found more than 1 architecture files!')
    arch_f0 = arch_f0[0]
    with open(arch_f0) as fp:
        arch_f0 = json.load(fp)


    features = read_whole_features(args.file_pattern.format(args.src))

    f0_cwt = features['lf0_cwt_norm']
    f0_cwt = nh_to_nhwc(f0_cwt)


    y_s_f0 = features['speaker']
    y_t_id_f0 = tf.placeholder(dtype=tf.int64, shape=[1,])
    y_t_f0 = y_t_id_f0 * tf.ones(shape=[tf.shape(f0_cwt)[0],], dtype=tf.int64)
    if not os.path.isdir('./f0_results'):
        os.mkdir('./f0_results')


# convert f0:
    machine_f0 = MODEL(arch_f0, is_training=False)
    z_f0 = machine_f0.encode(f0_cwt)
    f0_cwt_t = machine_f0.decode(z_f0, y_t_f0)  # NOTE: the API yields NHWC format
    f0_cwt_t = tf.squeeze(f0_cwt_t)

    output_dir = get_default_output(args.output_dir)
    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=output_dir)
    with sv.managed_session() as sess:
        load(saver, sess, logdir_f0, ckpt=ckpt_f0_cwt)
        print()
        while True:
            try:
                feat, lf0_cwt = sess.run(
                    [features, f0_cwt_t],
                    feed_dict={y_t_id_f0: np.asarray([SPEAKERS.index(args.trg)])}
                )
                feat.update({'lf0_cwt': lf0_cwt})

                feats = dic2npy(feat)
                oFilename = make_output_bin_name(output_dir, feat['filename'])

                with open(join('./f0_results', '{}.bin'.format(oFilename)), 'wb') as fp:
                    fp.write(feats.tostring())

            except KeyboardInterrupt:
                break
            finally:
                pass
        print()

if __name__ == '__main__':
    tf.app.run()
