__author__ = 'nglaz'

import json
import numpy as np
import os
from json_generator import generate_region_data

default_beats_folder = '../data/frames/'
default_result_folder = '../result/'


def generate_loop(beats_folder, result_folder):
    '''
    Generates a loop from the bars 5-6 (beats 16 to 24)
    :param beats_folder:
    :param result_folder:
    :return:
    '''
    for f in os.listdir(beats_folder):
        file_path = os.path.join(beats_folder, f)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            base_name = f[:f.index('_')]
        beat_frames = np.genfromtxt(file_path, dtype=int)
        dummy_loop = beat_frames[16:24]
        result_file = os.path.join(result_folder, base_name + '.json')
        data = generate_region_data( [ (dummy_loop[0], dummy_loop[-1]) ] )
        with open(result_file, 'wb') as output_file:
            json.dump(data, output_file)
        print f


if __name__ == '__main__':
    generate_loop(default_beats_folder, default_result_folder)