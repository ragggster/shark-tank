from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import os
import dill as pickle

# https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction

def extract_mfcc_features(filename):
	[Fs, x] = audioBasicIO.readAudioFile(filename)
	x = audioBasicIO.stereo2mono(x)
	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)

	# F is a matrix of (34, 1940), for example
	# N = 1940, where N is the number of short-term frames that fit into the input audio recording

	mfcc = F[8:21,:] # 13 mfcc features
	mfcc_mean = np.mean(mfcc, axis=1)
	mfcc_var = np.var(mfcc, axis=1)
	mfcc_min = np.amin(mfcc, axis=1)
	mfcc_max = np.amax(mfcc, axis=1)

	mfcc_stats = np.concatenate((mfcc_mean, mfcc_var, mfcc_min, mfcc_max), axis=0)
	mfcc_stats = np.reshape(mfcc_stats, (1, mfcc_stats.shape[0]))
	return mfcc_stats

# read data file season4_data.txt to get wav file names and corresponding labels
def readDataFile(filename):
	open_=open(filename,"r")
	lines=open_.readlines()
	wav_names=[]
	labels = []
	for line in lines:
		split_ = line.split()
		wav_names.append(split_[0])
		labels.append(split_[1])

	return (wav_names, labels)

# build feature matrix for one season
def getXandYForSeason(season):
	pitch_dir = './audio-scraping/season%s-pitches' %(season)
	data_file = 'data/season%s_data.txt' %(season) #may need to remove
	wav_names, labels = readDataFile(data_file)

	all_features = None
	for wav in wav_names:
		full_wav = pitch_dir + "/" + wav + ".wav"
		print(full_wav)
		assert(os.path.isfile(full_wav))
		feats = extract_mfcc_features(full_wav)
		if all_features is not None:
			all_features = np.concatenate((all_features, feats), axis=0)
		else:
			all_features = feats
	return all_features


def generateXandYForAllData():
	seasons = [4, 5, 8]
	all_X = None
	all_y = None
	for season in seasons:
		X_season, y_season = getXandYForSeason(season)
		if all_X is not None:
			all_X = np.concatenate((all_X, X_season), axis=0)
			all_y = np.concatenate((all_y, y_season), axis=0)
		else:
			all_X = X_season
			all_y = y_season
	'''season = 4
	pitch_dir = './audio-scraping/season%s-pitches' %(season)
	data_file = 'data/season%s_data.txt' %(season)
	wav_files = os.listdir(pitch_dir)

	wav_names, labels = readDataFile(data_file)
	
	all_features = None
	for wav in wav_names:
		full_wav = pitch_dir + "/" + wav + ".wav"
		print(full_wav)
		assert(os.path.isfile(full_wav))
		feats = extract_mfcc_features(full_wav)
		if all_features is not None:
			all_features = np.concatenate((all_features, feats), axis=0)
		else:
			all_features = feats
	print(all_features.shape)'''

	return (all_X, all_y)

#http://www.scipy-lectures.org/advanced/scikit-learn/

def classifyPitches():
	X, y = generateXandYForAllData()
	# split data into train/val 
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
	classifier = svm.LinearSVC()
	classifier.fit(X_train, y_train)
	classifier.predict(X_val)
	print("Accuracy is:")
	print(classifier.score(X_val, y_val))


createWeightMatrixFromPitches()




