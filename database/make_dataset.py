# this creates the ground truth data

import numpy as np

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

file_db = np.genfromtxt("audio_files.csv", delimiter=",", dtype=object)
c = np.genfromtxt("db/C.csv", delimiter=",", dtype=object)

for fn in c:
	if fn == 'T027': continue
	if fn == 'T078': continue
	# beattimes = np.genfromtxt("../data/aubiotempo/" + fn + "_vamp_vamp-aubio_aubiotempo_beats.csv", delimiter=",")
	beattimes = np.genfromtxt("/Users/hvkoops/repos/kanyAI/data/beatroot/" + fn + "_vamp_beatroot-vamp_beatroot_beats.csv", delimiter=",")
	csvcontents = np.zeros((len(beattimes), 6))
	samplestarts = np.genfromtxt("features/" + fn + "_starttimes.csv", delimiter=",")
	if samplestarts.ndim > 1:
		samplestarts_s = samplestarts[:,1]
		samplestarts_fr = samplestarts[:,0]
	if samplestarts.ndim == 1:
		samplestarts_s = np.array([samplestarts[1]])
		samplestarts_fr = np.array([samplestarts[0]])
	# find nearest beat of start time:
	for bt in np.arange(len(beattimes)):
		for s in samplestarts_s:
			nearest = find_nearest(beattimes, s)
			if beattimes[bt] == nearest:
				this =  np.array([beattimes[bt], 1, 0, 0, 0, 0])
			else:
				this = np.array([beattimes[bt], 0, 0, 0, 0, 0])
			csvcontents[bt] = this
	for ones in np.where(csvcontents[:,1]==1)[0]:
		csvcontents[:,2][ones:ones+4] = np.ones(4)[0:len(csvcontents)-ones]
		csvcontents[:,3][ones:ones+8] = np.ones(8)[0:len(csvcontents)-ones]
		csvcontents[:,4][ones:ones+16] = np.ones(16)[0:len(csvcontents)-ones]
		csvcontents[:,5][ones:ones+32] = np.ones(32)[0:len(csvcontents)-ones]

	# print csvcontents
	csvfname = "groundtruth/" + fn + "_groundtruth_beatroot.csv"
	np.savetxt(csvfname, csvcontents, delimiter=",")
	print "wrote: " + csvfname
