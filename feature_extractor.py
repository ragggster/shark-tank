
import python_speech_features
import tensorflow as tf 
import numpy as np
import scipy.io.wavfile as wav
import os
from os import listdir
from os.path import isfile, join, exists
import wave
from scikits.audiolab import Sndfile, play
from scikits import audiolab
from unpickle import unpickle
import shutil

SEASONS = [8, 4]
DOWNSAMPLE = 1000
SPLIT_TIME = 20 #in seconds
MIN_SIZE = 250000 #in number of samples, 250000 is about 6 seconds

MFCC_DIR = './data/mfcc' 
LABELS_DIR = './data/labels/'

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if (isfile(join(directory, f)) and f[0] != '.')]

def get_label(line):
	potential_labels = line.split(' ')[0] #ALTER THE INDEX TO CHANGE WHAT YOU ARE PREDICTING!!!!
	return float(potential_labels)

class MFCC_Extractor():
	def __init__(self, season, split_time = 20):
		self.data_dir = './audio-scraping/season%s-pitches' %(season)
		self.mfcc_dir = MFCC_DIR
		self.meta_file = './audio-scraping/season%s-pitches-metadata.p' % (season)
		self.labels_dir = LABELS_DIR
		self.input_fns = get_files_in_dir(self.data_dir)
		self.split_time = split_time #should be the number of seconds to put in each one


	def split_episodes(self, signal, rate):
		seg_len = self.split_time*rate
		for seg_start in range(len(signal))[::seg_len]:
			seg_end = min([len(signal), seg_start+seg_len])
			l = seg_end - seg_start
			if l < MIN_SIZE:
				continue
			print 'split time: %f seconds (%i samples)' %(1.0*l/rate, l)
			yield signal[seg_start:seg_end]

	def write_features(self):
		# Write features as csv WITH headers so we can pick which to classify later
		# load data
		
		for pitch_audio_fn in self.input_fns:
			
			full_file = join(self.data_dir, pitch_audio_fn)
			open_wav = Sndfile(full_file)
			rate = open_wav.samplerate#*DOWNSAMPLE
			
			sig = open_wav.read_frames(open_wav.nframes)
			
			for i, split in enumerate(self.split_episodes(sig, rate)):
				mfcc_features = python_speech_features.mfcc(split, rate)
				with open(join(self.mfcc_dir, pitch_audio_fn.split('.')[0] + ".%i" %(i)), 'w')	as output_fn:
					np.savetxt(output_fn, mfcc_features, delimiter= ',')

					print "Extracted MFCC features for %s, split %i, into: %s" %(pitch_audio_fn, i, output_fn.name)
if __name__ == '__main__':
	if exists(MFCC_DIR): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST! NB
		shutil.rmtree(MFCC_DIR)
	os.makedirs(MFCC_DIR)

	for season in SEASONS:
		MFCC_Extractor(season).write_features()
