from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os
import dill as pickle

''' 
DESCRIPTION
------------
Binary SVM classifier that classifies pitches as funded/not funded.
Note that currently only MFCC features are being used.

So far performance is shitty; we need more data
'''

# gets the statistics of the mfcc vectors for a given pitch
# input: np array of mfcc vector for each time step in wav file
def get_mfcc_stats(mfcc):
	mfcc_mean = np.mean(mfcc, axis=1)
	mfcc_var = np.var(mfcc, axis=1)
	mfcc_median = np.var(mfcc, axis=1)
	mfcc_min = np.amin(mfcc, axis=1)
	mfcc_max = np.amax(mfcc, axis=1)
	mfcc_stats = np.concatenate((mfcc_mean, mfcc_var, mfcc_median, mfcc_min, mfcc_max), axis=0)
	mfcc_stats = np.reshape(mfcc_stats, (1, mfcc_stats.shape[0]))

	return mfcc_stats

#build feature matrix for one season, currently just MFCCs
def getXandYForSeason(season):
	# Unpickle the season's labels
	pickled_labels_file = './audio-scraping/season%s-labelled.p' %(season)
	f = open(pickled_labels_file, 'r')
	label_dict = pickle.load(f)
	f.close()

	# Unpickle the season's mfcc vectors
	pickled_mfcc_file = './svm-features/season%s-mfcc.p' %(season)
	f2 = open(pickled_mfcc_file, 'r')
	mfcc_dict = pickle.load(f2)
	f2.close()

	X_season = None
	y_season = []

	for pitch_key, pitch_labels in label_dict.iteritems():
		pitch_key = os.path.splitext(pitch_key)[0] #remove .wav
		#print(pitch_key)

		# compute mfcc statistics
		mfcc_stats = get_mfcc_stats(mfcc_dict[pitch_key])
		label = pitch_labels['label_code'] # 1 or 0 funded/not funded
		if X_season is not None:
			X_season = np.vstack((X_season, mfcc_stats))
		else:
			X_season = mfcc_stats

		y_season.append(label)

	y_season = np.array(y_season)
	return (X_season, y_season)

def generateXandYForAllSeasons():
	seasons = [4, 5]
	all_X = None
	all_y = None

	for season in seasons:
		X_season, y_season = getXandYForSeason(season)
		if all_X is not None:
			all_X = np.vstack((all_X, X_season))
			all_y = np.hstack((all_y, y_season))
		else:
			all_X = X_season
			all_y = y_season

	return (all_X, all_y)

def classifyPitches():
	X, y = generateXandYForAllSeasons()
	print(X.shape)
	# split data into train/val 
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0) #optional param: random_state=31
	classifier = svm.LinearSVC()
	classifier.fit(X_train, y_train)
	classifier.predict(X_val)
	print("Accuracy is:")
	print(classifier.score(X_val, y_val))

def crossValidate():
	X, y = generateXandYForAllSeasons()
	classifier = svm.LinearSVC()
	scores = cross_val_score(classifier, X, y, cv=5) #can change the fold number
	print(scores)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#classifyPitches()
crossValidate()





