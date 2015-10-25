import numpy as np
import os

from aggregate import hop_size, sr

input_folder = '../data/qm-tempotracker/'
#input_folder = '../data/beatroot/'
output_folder = '../data/frames/'


def main():
    seconds_per_frame = float(hop_size) / sr
    for f in os.listdir(input_folder):
        file_path = os.path.join(input_folder, f)
        if os.path.isfile(file_path) and file_path.endswith('.csv'):
            base_name = f[:f.index('_')]
            data = np.genfromtxt(file_path, usecols=(0), delimiter=',').T
            frames = [int(round(seconds / seconds_per_frame)) for seconds in data]
            with open(os.path.join(output_folder, '%s_beats.csv' % base_name), 'w') as output_file:
                for frame in frames:
                    print >>output_file, frame
            print 'Done ' + f


if __name__ == '__main__':
    main()