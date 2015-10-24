import numpy as np

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

file_db = np.genfromtxt("audio_files.csv", delimiter=",", dtype=object)
c = np.genfromtxt("db/C.csv", delimiter=",", dtype=object)

for fn in c:
	beattimes = np.genfromtxt("../data/aubiotempo/" + fn + "_vamp_vamp-aubio_aubiotempo_beats.csv", delimiter=",")
	samplestarts = np.genfromtxt("features/" + fn + "_starttimes.csv", delimiter=",")
	if samplestarts.ndim > 1:
		samplestarts_s = samplestarts[:,0]
		samplestarts_fr = samplestarts[:,1]
	if samplestarts.ndim == 1:
		samplestarts_s = samplestarts[0]
		samplestarts_fr = samplestarts[1]
	# find nearest beat of start time:
	find_nearest(beattimes, samplestarts_s)