__author__ = 'Jan'

import numpy as np
import librosa as lr

sr = 44100
hop_size = 1024
beat_folder = '../data/aubiotempo/'

def beat_aggregation(feature_data, file_name, time_col=False, agg_function=np.median):
    feat_frames = np.genfromtxt(feature_data).T
    if time_col:
        feat_frames = feat_frames[1:]
    beat_times = np.genfromtxt(beat_folder + file_name + '_vamp_vamp-aubio_aubiotempo_beats.csv')
    beat_frames = beat_times * float(sr)/hop_size

    feat_beats = lr.feature.sync(feat_frames, beat_frames, aggregate=agg_function)
    np.savetxt(feature_data.split('.')[-2] + '_beat.csv', feat_beats.T, delimiter=',')