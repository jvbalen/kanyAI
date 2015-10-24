# This python file reads the sample database

import numpy as np

# First sheet in the DB
track_db = np.genfromtxt("sample_database.csv", delimiter=";", dtype=object)
# Second sheet in the DB
sample_db = np.genfromtxt("sample_database2.csv", delimiter=";", dtype=object)
# List of audio files in the dataset
file_db = np.genfromtxt("audio_files.csv", dtype=object)

# contents of the first sheet in sample_database.xls
tracks_trackids = track_db[:,0]
tracks_artists = track_db[:,1]
tracks_titles = track_db[:,2]
tracks_years = track_db[:,3].astype(int)
tracks_genres = track_db[:,4]

# contents of the second sheet in sample_database.xls
sample_sampleids = sample_db[:,0]
sample_sampling_track = sample_db[:,1]
sample_track_sampled = sample_db[:,2]
sample_interpolation = sample_db[:,3]
sample_time_sampled =  sample_db[:,4]
sample_times_of_sample = sample_db[:,5]
sample_repetitions = sample_db[:,6]
sample_type_loop = sample_db[:,7]
sample_comments = sample_db[:,8]

# for i in np.arange(0,len(trackids)):
# 	if not trackids[i] in file_db:
# 		clean_db = np.delete(track_db, (i), axis=0)