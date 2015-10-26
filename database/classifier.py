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

feature_extraction = 0
machine_learning = 1
predicting = 1

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
	c = np.genfromtxt("db/C.csv", delimiter=",", dtype=object)

	allclasses = []
	allfeatures = []
	# read the feature csv file
	for fn in c:		
		classfile = "groundtruth/"+fn+"_groundtruth_beatroot.csv"
		classes = np.genfromtxt(classfile, delimiter=",")
		classes1 = classes[:,1]
		classes4 = classes[:,2]
		classes8 = classes[:,3]
		classes16 = classes[:,4]
		classes32 = classes[:,5]

		featurefile_cq = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated/"+fn+".csv"
		featurefile_std = "features/vincent_beats_std/"+fn+".csv"
		featurefile_med = "features/vincent_beats_median/"+fn+".csv"
		featurefile_en = "features/vincent_beats_entropy/"+fn+".csv"
		constantqagg = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated/"+fn+".csv"
		constantqaggentropy = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated-entropy/"+fn+".csv"
		constantqaggstd = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated-std/"+fn+".csv"

		features_cq = np.genfromtxt(featurefile_cq, delimiter=",")
		features_std = np.genfromtxt(featurefile_std, delimiter=",")
		features_med = np.genfromtxt(featurefile_med, delimiter=",")
		features_en = np.nan_to_num(np.genfromtxt(featurefile_en, delimiter=","))
		features_cqagg = np.genfromtxt(constantqagg, delimiter=",")
		features_cqen = np.nan_to_num(np.genfromtxt(constantqaggentropy, delimiter=","))
		features_cqstd = np.genfromtxt(constantqaggstd, delimiter=",")

		# std_mfcc  = features_std[:,175:187]
		# med_mfcc  = features_med[:,175:187]

		# features = med_mfcc
		# features = np.hstack((std_mfcc, med_mfcc))
		features = np.hstack((features_cq,features_std,features_med,features_en,features_cqagg,features_cqen,features_cqstd))

		# equal sampling:
		barlen = 8
		no_0barsseq = np.where(classes1==0)[0]
		no_1barsseq = np.where(classes1==1)[0]
		for nb in no_1barsseq:
			cl_pos_pos = (nb,nb+barlen)
			cl_pos_ex = np.ones(barlen)
			randpos0 = np.random.randint(1,len(np.where(classes1==0)[0])-barlen)
			cl_neg_pos = (randpos0,randpos0+barlen)
			cl_neg_ex = np.zeros(barlen)
			allclasses.append(cl_pos_ex)
			allclasses.append(cl_neg_ex)

			fe_pos_ex = features[cl_pos_pos[0]:cl_pos_pos[1]]
			fe_neg_ex = features[cl_neg_pos[0]:cl_neg_pos[1]]
			allfeatures.append(fe_pos_ex)
			allfeatures.append(fe_neg_ex)

	allclasses = np.concatenate(allclasses)	
	allfeatures = np.concatenate(allfeatures)	

	test_size = 0.3
	clf_rand_state = np.random.randint(100, size=1)[0]
	clf = Pipeline([
		('feature_selection', LinearSVC(penalty="l1", dual=False)),
		('classification', GradientBoostingClassifier(n_estimators=700, learning_rate=0.1, max_depth=3, random_state=0, verbose=1))
		])

	# random_state = np.random.randint(100, size=1)[0]
	# X_trainr, X_testr, y_trainr, y_testr = cross_validation.train_test_split(allfeatures, allclasses, test_size=test_size, random_state=0)		
	X_train, X_test, y_train, y_test = allfeatures[0:761], allfeatures[762:], allclasses[0:761], allclasses[762:]		
	print "Fitting model: " + str(i)
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print "Accuracy: " + str(score)

	clf.fit(allfeatures, allclasses)

if predicting:
	# for all test files:
	for fn in glob("/Users/hvkoops/repos/kanyAI//data/test_audio/wav/*.wav"):
		thisf = fn[49:-4]

		te_featurefile_cq = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated/"+thisf+".csv"
		te_featurefile_std = "/Users/hvkoops/repos/kanyAI/data/test_features/vincent_beats_std/"+thisf+".csv"
		te_featurefile_med = "/Users/hvkoops/repos/kanyAI/data/test_features/vincent_beats_median/"+thisf+".csv"
		te_featurefile_en = "/Users/hvkoops/repos/kanyAI/data/test_features/vincent_beats_entropy/"+thisf+".csv"
		te_constantqagg = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated/"+thisf+".csv"
		te_constantqaggentropy = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated-entropy/"+thisf+".csv"
		te_constantqaggstd = "/Users/hvkoops/repos/kanyAI/data/constantq-aggregated-std/"+thisf+".csv"

		te_features_cq = np.nan_to_num(np.genfromtxt(te_featurefile_cq, delimiter=",")).astype(np.float32)
		te_features_std = np.nan_to_num(np.genfromtxt(te_featurefile_std, delimiter=",")).astype(np.float32)
		te_features_med = np.nan_to_num(np.genfromtxt(te_featurefile_med, delimiter=",")).astype(np.float32)
		te_features_en = np.nan_to_num(np.genfromtxt(te_featurefile_en, delimiter=",")).astype(np.float32)
		te_features_cqagg = np.nan_to_num(np.genfromtxt(te_constantqagg, delimiter=",")).astype(np.float32)
		te_features_cqen = np.nan_to_num(np.genfromtxt(te_constantqaggentropy, delimiter=",")).astype(np.float32)
		te_features_cqstd = np.nan_to_num(np.genfromtxt(te_constantqaggstd, delimiter=",")).astype(np.float32)

		te_features = np.hstack((te_features_cq,te_features_std,te_features_med,np.nan_to_num(te_features_en),te_features_cqagg,te_features_cqen,te_features_cqstd))

		print "Testing score..."
		predictions = clf.predict(te_features)
		predictions_out = predictions.reshape((predictions.shape[0],1))
		np.savetxt(thisf+"_prediction.csv", predictions_out)

print "- Done"	