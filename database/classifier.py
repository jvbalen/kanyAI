# binary classifier for sample detection

import matplotlib.pyplot as plt
import itertools
from yaafelib import *
from math import * 

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC

import numpy as np

feature_extraction = 1
machine_learning = 0

# # ================================= CSV READING ============================================ 
# db = np.genfromtxt("db/DB.csv", delimiter=",", dtype=object)
# # db fields
# db_sampleids = db[:,0]
# db_sampling_track = db[:,1] # q
# db_track_sampled = db[:,2] # c
# db_interpolation = db[:,3]
# db_time_sampled =  db[:,4]
# db_times_of_sample = db[:,5]
# db_repetitions = db[:,6]
# db_type_loop = db[:,7]

# all filenames
file_db = np.genfromtxt("audio_files.csv", delimiter=",", dtype=object)

# ================================= FEATURE EXTRACTION ============================================ 
if feature_extraction:
	# feature extraction parameters:
	sr=44100
	blockSize=str(1024)
	stepSize=str(1024)
	featureplan = [	"am: AmplitudeModulation blockSize="+blockSize+" stepSize="+stepSize,
					"ac: AutoCorrelation ACNbCoeffs=49 blockSize="+blockSize+" stepSize="+stepSize,
					"cdod: ComplexDomainOnsetDetection FFTLength=0  FFTWindow=Hanning blockSize="+blockSize+" stepSize="+stepSize,
					"en: Energy blockSize="+blockSize+" stepSize="+stepSize,
					"env: Envelope EnDecim=200 blockSize="+blockSize+" stepSize="+stepSize,
					"envsh: EnvelopeShapeStatistics EnDecim=200  blockSize="+blockSize+" stepSize="+stepSize,
					"lpc: LPC LPCNbCoeffs=4  blockSize="+blockSize+" stepSize="+stepSize,
					"lfs: LSF LSFDisplacement=1  LSFNbCoeffs=10 blockSize="+blockSize+" stepSize="+stepSize,
					"lx: Loudness blockSize="+blockSize+" stepSize="+stepSize,
					"mfcc: MFCC blockSize="+blockSize+" stepSize="+stepSize+" CepsNbCoeffs=16",
					# "ms: MagnitudeSpectrum FFTLength=0 FFTWindow=Hanning blockSize="+blockSize+" stepSize="+stepSize,
					"obsi: OBSI blockSize="+blockSize+" stepSize="+stepSize,
					"obsir: OBSIR blockSize="+blockSize+" stepSize="+stepSize,
					"psh: PerceptualSharpness blockSize="+blockSize+" stepSize="+stepSize,
					"psp: PerceptualSpread blockSize="+blockSize+" stepSize="+stepSize,
					"scfpb: SpectralCrestFactorPerBand blockSize="+blockSize+" stepSize="+stepSize,
					"sd: SpectralDecrease blockSize="+blockSize+" stepSize="+stepSize,
					"sf: SpectralFlatness blockSize="+blockSize+" stepSize="+stepSize,
					"sfpb: SpectralFlatnessPerBand FFTLength=0 FFTWindow=Hanning blockSize="+blockSize+" stepSize="+stepSize,					
					"sfx: SpectralFlux FFTLength=0  FFTWindow=Hanning  FluxSupport=All  blockSize="+blockSize+" stepSize="+stepSize,					
					"srl: SpectralRolloff FFTLength=0  FFTWindow=Hanning   blockSize="+blockSize+" stepSize="+stepSize,					
					"sss: SpectralShapeStatistics blockSize="+blockSize+" stepSize="+stepSize,
					"ss: SpectralSlope blockSize="+blockSize+" stepSize="+stepSize,
					"sv: SpectralVariation blockSize="+blockSize+" stepSize="+stepSize,
					"tss: TemporalShapeStatistics blockSize="+blockSize+" stepSize="+stepSize,
					"zcr: ZCR blockSize="+blockSize+" stepSize="+stepSize
					]
	fp = FeaturePlan(sample_rate=sr, normalize=True, resample=True)
	for x in xrange(0,len(featureplan)):
		fp.addFeature(featureplan[x])

	df = fp.getDataFlow()
	engine = Engine()
	engine.load(df)
	afp = AudioFileProcessor()

	# read all files in file_db:
	for fn in file_db:
		wavname = "../audio/wav/" + fn + ".wav"
		afp.processFile(engine,wavname)
		feats = engine.readAllOutputs()	
		thisfeats = np.concatenate(feats.values(), axis=1)
		np.save("features/" + fn + "_feats.npy", thisfeats)
		np.savetxt("features/" + fn + "_feats.csv", thisfeats, delimiter=",")


# ================================= MACHINE LEARNING ============================================ 
if machine_learning:
	# read the feature csv file
	featurefile = "features.csv"
	size = 300
	nfeats = 17
	# features = np.genfromtxt(featurefile, delimiter=",")
	features = np.random.rand(size,nfeats)

	classfile = "classes.csv"
	# classes = np.genfromtxt(classfile, delimiter=",")
	classes = np.random.randint(2, size=size)
	test_size = 0.3

	clf_rand_state = np.random.randint(100, size=1)[0]
	clf = Pipeline([
		('classification', GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=clf_rand_state))
		])

	# k repeats
	k = 10
	scores = np.zeros(k)
	for i in np.arange(k):
		random_state = np.random.randint(100, size=1)[0]
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, classes, test_size=test_size, random_state=random_state)
		print "Fitting model: " + str(i)
		clf.fit(X_train, y_train)
		print "Testing score..."
		score = clf.score(X_test, y_test)
		print "Accuracy: " + str(score)
		scores[i] = score

	avg_acc = np.mean(scores)
	print "Scores: " + str(scores)
	print "Average accuracy: " + str(avg_acc)

print "- Done"	