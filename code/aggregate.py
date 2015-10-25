__author__ = 'Jan'

import numpy as np
import librosa as lr
import os

sr = 44100
hop_size = 1024
beat_file_suffix = '_beats.csv'
default_beats_folder = '../data/frames/'
default_features_folder = '../data/constantq/'


def aggregate_vamp_features(features_folder, beats_folder, time_col=True, agg_function=np.median, output_folder=None):
    '''
    Aggregates feature matrix for each file using the beat frame numbers given in the files in beats_folder
    using the provided agg_function and librosa.feature.sync. It is supposed that the feature matrix has been
    calculated using sr=44100 and hop_size=1024/
    :param features_folder:
    :param beats_folder:
    :param time_col: whether the 1st column of feature matrix contains time. Default: True.
    :param agg_function: default: np.median
    :param output_folder: default: features_folder + '-aggregated/'
    :return:
    '''
    if not output_folder:
        output_folder = features_folder.rstrip('/') + '-aggregated/'
    for f in os.listdir(features_folder):
        file_path = os.path.join(features_folder, f)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            base_name = f[:f.index('_')]
            beat_file_path = beats_folder + base_name + beat_file_suffix
            print file_path
            features_data = np.genfromtxt(file_path, delimiter=',').T
            if time_col:
                features_data = features_data[1:]
            beat_frames = np.genfromtxt(beat_file_path, dtype=int)
            feat_beats = lr.feature.sync(features_data, beat_frames, aggregate=agg_function)
            np.savetxt(output_folder + base_name + '.csv', feat_beats.T, delimiter=',', fmt='%.8f')


if __name__ == '__main__':
    aggregate_vamp_features(default_features_folder, default_beats_folder)
