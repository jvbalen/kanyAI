# extract test file features

import matplotlib.pyplot as plt
import itertools
from yaafelib import *
from math import * 

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from glob import glob

import numpy as np

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
for fn in glob("../data/test_audio/wav/*.wav"):
	afp.processFile(engine,fn)
	feats = engine.readAllOutputs()	
	thisfeats = np.concatenate(feats.values(), axis=1)
	np.save(fn + "_feats.npy", thisfeats)
	np.savetxt(fn + "_feats.csv", thisfeats, delimiter=",")