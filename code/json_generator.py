__author__ = 'nglaz'

import os
import numpy as np
import json
from aggregate import sr, hop_size

default_top_predictions_folder = '../data/top-predictions/'
default_frames_folder = '../data/frames/'
default_result_folder = '../result/'


def generate_region_data(frame_number_pairs):
    '''
    Generates the region data objects as required by wavesurfer.regions.js
    :param frame_number_pairs: list of frame number pairs (region_start, region_end)
    :return: object that can be directly dumped to JSON for wavesurfer.regions.js
    '''
    region_data = []
    seconds_per_frame = float(hop_size) / sr
    for pair in frame_number_pairs:
        region_data.append({'start': (pair[0] * seconds_per_frame),
                'end': (pair[1] * seconds_per_frame),
                'attributes': {'label': 'Loop 0', 'highlight': True},
                'data': {}})
    return region_data


def main(top_predictions_folder, frames_folder, result_folder):
    seconds_per_frame = float(hop_size) / sr
    for f in os.listdir(top_predictions_folder):
        print f
        file_path = os.path.join(top_predictions_folder, f)
        top_frames = np.genfromtxt(file_path, dtype=int)
        base_name = f[:f.rindex('_')]
        frames_file = frames_folder + base_name + '_beats.csv'
        beat_frames = np.genfromtxt(frames_file, dtype=int)
        pairs = []
        for tf in top_frames:
            i = 0
            while beat_frames[i] < tf:
                i += 1
            pair = (beat_frames[i], beat_frames[i+8])
            pairs.append(pair)
        result_file = os.path.join(result_folder, base_name + '.json')
        data = generate_region_data(set(pairs))
        with open(result_file, 'wb') as output_file:
            json.dump(data, output_file)


if __name__ == '__main__':
    main(default_top_predictions_folder, default_frames_folder, default_result_folder)