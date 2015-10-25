import numpy as np
import os

from aggregate import hop_size, sr, beat_file_suffix

default_input_folder = '../data/qm-tempotracker/'
#default_input_folder = '../data/beatroot/'
default_output_folder = '../data/frames/'


def seconds_to_frames(input_folder, output_folder):
    '''
    Converts a sequence of time values from the first column of the input file to a sequence of frame numbers
     and saves this sequence of the ouptut file
    :param input_folder:
    :param output_folder:
    :return:
    '''
    seconds_per_frame = float(hop_size) / sr
    for f in os.listdir(input_folder):
        file_path = os.path.join(input_folder, f)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            base_name = f[:f.index('_')]
            data = np.genfromtxt(file_path, usecols=(0), delimiter=',').T
            frames = [int(round(seconds / seconds_per_frame)) for seconds in data]
            np.savetxt(os.path.join(output_folder, base_name + beat_file_suffix), frames, fmt='%d')
            print 'Done ' + f


if __name__ == '__main__':
    seconds_to_frames(default_input_folder, default_output_folder)