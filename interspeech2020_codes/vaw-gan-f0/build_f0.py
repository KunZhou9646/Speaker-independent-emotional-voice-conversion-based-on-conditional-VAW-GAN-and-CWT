import numpy as np
import pyworld as pw
import soundfile as sf
import tensorflow as tf
from analyzer_f0 import read_f0, read_whole_features, norm_scale,get_cont_lf0,convert_continuos_f0


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('corpus_name', 'emotion_vc', 'Corpus Name')
tf.app.flags.DEFINE_string(
    'speaker_list', './etc/speakers.tsv', 'Speaker list (one speaker per line)'
)
tf.app.flags.DEFINE_string(
    'train_file_pattern',
    './data_multi/bin/training_set/*/*.bin',
    'training dir (to *.bin)'
)

def main():
    tf.gfile.MkDir('./etc')

    # ==== Save max and min value ====
    x = read_whole_features(args.train_file_pattern)  # TODO: use it as a obj and keep `n_files`
    y_all = list()
    lf0_cwt_all = list()
    f0_all = list()
    lf0_cwt_norm_all = list()
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        counter = 1
        while True:   # TODO: read according to speaker instead of all speakers
             try:
                features = sess.run(x)
                print('\rProcessing {}: {}'.format(counter, features['filename']), end='')
                y_all.append(features['speaker'])
                lf0_cwt_all.append(features['lf0_cwt'])
                f0_all.append(features['f0'])
                lf0_cwt_norm_all.append(features['lf0_cwt_norm'])
                counter += 1
             finally:
                pass
        print()

    y_all = np.concatenate(y_all,axis=0)
    lf0_cwt_all = np.concatenate(lf0_cwt_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)
    lf0_cwt_norm_all = np.concatenate(lf0_cwt_norm_all, axis=0)


    with open(args.speaker_list) as fp:
        SPEAKERS = [l.strip() for l in fp.readlines()]

    # ==== F0 stats ====
    for s in SPEAKERS:
        print('Speaker {}'.format(s), flush=True)
        f0 = f0_all[SPEAKERS.index(s) == y_all]
        #f0 = f0[f0 > 2.]
        _, lf0_cont = get_cont_lf0(f0)
        #f0 = np.log(f0)
        mu, std = lf0_cont.mean(), lf0_cont.std()

        # Save as `float32`
        with open('./etc/{}.npf'.format(s+'f0'), 'wb') as fp:
            fp.write(np.asarray([mu, std]).tostring())

    # ==== lf0_cwt stats ====
    for s in SPEAKERS:
        print('Speaker {}'.format(s+'_lf0_cwt'), flush=True)
        lf0_cwt = lf0_cwt_all[SPEAKERS.index(s) == y_all]
        lf0_cwt_norm, mu_lf0_cwt, std_lf0_cwt = norm_scale(lf0_cwt)
        mu_lf0_cwt = mu_lf0_cwt.T
        mu_lf0_cwt = mu_lf0_cwt.flatten()
        std_lf0_cwt = std_lf0_cwt.T
        std_lf0_cwt = std_lf0_cwt.flatten()
        # Save as `float32`
        with open('./etc/{}.npf'.format(s+'_lf0_cwt'), 'wb') as fp:
            fp.write(np.asarray([mu_lf0_cwt, std_lf0_cwt]).tostring())






if __name__ == '__main__':
    main()
