from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import neighbors
from os import listdir


# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction

def extract_mfcc_features():
	[Fs, x] = audioBasicIO.readAudioFile("s8-e1-p1.wav")
	x = audioBasicIO.stereo2mono(x)
	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)

	# F is a matrix of (34, 1940)
	# N = 1940, where N is the number of short-term frames that fit into the input audio recording

	mfcc = F[8:21,:] # 13 mfcc features
	mfcc_mean = np.mean(mfcc, axis=1)
	mfcc_var = np.var(mfcc, axis=1)
	mfcc_min = np.amin(mfcc, axis=1)
	mfcc_max = np.amax(mfcc, axis=1)

	mfcc_stats = np.concatenate((mfcc_mean, mfcc_var, mfcc_min, mfcc_max), axis=0)
	print(mfcc_stats)
	return mfcc_stats

def createWeightMatrixFromPitches():
	season = 4
	pitch_dir = './audio-scraping/season%s-pitches' %(season)
	wav_files = listdir(pitch_dir)
	print(wav_files)


#http://www.scipy-lectures.org/advanced/scikit-learn/

def classifyPitch():
	classifier = svm.LinearSVC()


createWeightMatrixFromPitches()




