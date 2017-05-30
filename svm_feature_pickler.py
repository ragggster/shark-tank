from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioBasicIO
import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import os
from os.path import join
import pickle

'''
Description
-----------
This python file extracts the mfcc features for every frame of a pitch wave file
and pickles a dictionary that maps a wav file to the corresponding mfcc numpy array

Resulting pickle files are stored in the svm-features directory

'''

def create_feature_pickle_dir():
	new_features_dir = './svm-features'
	if not os.path.exists(new_features_dir):
			os.makedirs(new_features_dir)

def extract_mfcc_features(filename):
	print(filename)
	[Fs, x] = audioBasicIO.readAudioFile(filename)
	x = audioBasicIO.stereo2mono(x)
	F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
	mfcc = F[8:21,:] # 13 mfcc features
	return mfcc

def pickle_mfcc_features_for_season(season):
	season_pitches_dir = './audio-scraping/season%s-pitches' %(season)
	pickle_file = './svm-features/season%s-mfcc.p' %(season)
	mfcc_dict = {}

	wav_files = os.listdir(season_pitches_dir)
	for wav in wav_files:
		full_wav_path = join(season_pitches_dir, wav)
		pitch_key = os.path.splitext(wav)[0] #remove .wav
		mfcc = extract_mfcc_features(full_wav_path)
		mfcc_dict[pitch_key] = mfcc

	f = open(pickle_file, 'w')
	pickle.dump(mfcc_dict, f)
	f.close()


# Example: pickle season 5 mfcc
season_num = 5
create_feature_pickle_dir()
pickle_mfcc_features_for_season(season_num)
