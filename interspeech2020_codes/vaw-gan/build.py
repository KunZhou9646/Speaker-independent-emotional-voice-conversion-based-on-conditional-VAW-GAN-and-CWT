import numpy as np
import pyworld as pw
import soundfile as sf
import tensorflow as tf
from analyzer import pw2wav, read, read_whole_features


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('corpus_name', 'emotion_vc', 'Corpus Name')
tf.app.flags.DEFINE_string(
    'speaker_list', './etc/speakers.tsv', 'Speaker list (one speaker per line)'
)
tf.app.flags.DEFINE_string(
    'train_file_pattern',
    '../data/bin/training_set/*/*.bin',
    'training dir (to *.bin)'
)

def main():
    tf.gfile.MkDir('./etc')

    # ==== Save max and min value ====
    x = read_whole_features(args.train_file_pattern)  # TODO: use it as a obj and keep `n_files`
    x_all = list()
    y_all = list()
    f0_all = list()
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        counter = 1
        while True:   # TODO: read according to speaker instead of all speakers
            try:
                features = sess.run(x)
                print('\rProcessing {}: {}'.format(counter, features['filename']), end='')
                x_all.append(features['sp'])
                y_all.append(features['speaker'])
                f0_all.append(features['f0'])
                counter += 1
            finally:
                pass
        print()

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)

    with open(args.speaker_list) as fp:
        SPEAKERS = [l.strip() for l in fp.readlines()]

    # ==== F0 stats ====
    for s in SPEAKERS:
        print('Speaker {}'.format(s), flush=True)
        f0 = f0_all[SPEAKERS.index(s) == y_all]
        print('  len: {}'.format(len(f0)))
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()

        # Save as `float32`
        with open('./etc/{}.npf'.format(s), 'wb') as fp:
            fp.write(np.asarray([mu, std]).tostring())

    # ==== Min/Max value ====
    # mu = x_all.mean(0)
    # std = x_all.std(0)
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    # Save as `float32`
    with open('./etc/{}_xmin.npf'.format(args.corpus_name), 'wb') as fp:
        fp.write(q005.tostring())

    with open('./etc/{}_xmax.npf'.format(args.corpus_name), 'wb') as fp:
        fp.write(q995.tostring())



if __name__ == '__main__':
    main()
