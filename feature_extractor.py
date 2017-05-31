
import python_speech_features
import tensorflow as tf 
import numpy as np
import scipy.io.wavfile as wav
import os
from os import listdir
from os.path import isfile, join, exists
# import wave
# from scikits.audiolab import Sndfile, play
# from scikits import audiolab
import shutil
import dill as pickle
from collections import defaultdict

SEASONS = [5, 8, 4]
#DOWNSAMPLE = 1000
SPLIT_TIME = 10 #in seconds
MIN_SIZE = 250000 #in number of samples, 250000 is about 6 seconds

DATA_DIR = './data'
MFCC_DIR = DATA_DIR + '/mfcc' 

AUDIO_SCRAPING_DIR = "./audio-scraping/"

FINAL_LABEL_FILE = DATA_DIR + '/labels.p'

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if (isfile(join(directory, f)) and f[0] != '.')]

'''
For each season in the SEASONS variable, this script goes in, get's all of the 
wav files from the appropriate scraped audio folder, splits those files up 
into chunks of size defined by SPLIT_TIME, and extracts the mfcc features 
from that chunk. ALL data from all of the seasons are written to the 
same data folder. This script clears the mfcc data directory before starting
so you can run this repeatedly.

ALSO This compiles the label dictionaries into one dictionary
'''

class MFCC_Extractor():
	def __init__(self, season, split_time = SPLIT_TIME):
		self.data_dir = '%sseason%s-pitches' %(AUDIO_SCRAPING_DIR, season)
		self.mfcc_dir = MFCC_DIR
		self.meta_file = '%sseason%s-pitches-metadata.p' % (AUDIO_SCRAPING_DIR, season)
		self.input_fns = get_files_in_dir(self.data_dir)
		self.split_time = split_time #should be the number of seconds to put in each one

	def split_episodes(self, signal, rate):
		seg_len = self.split_time*rate
		for seg_start in range(len(signal))[::seg_len]:
			seg_end = min([len(signal), seg_start+seg_len])
			l = seg_end - seg_start
			if l < MIN_SIZE:
				continue
			print ('split time: %f seconds (%i samples)') %(1.0*l/rate, l)
			yield signal[seg_start:seg_end]

	def write_features(self):
		# Write features as csv WITH headers so we can pick which to classify later
		# load data
		
		for pitch_audio_fn in self.input_fns:
			
			full_file = join(self.data_dir, pitch_audio_fn)
			rate, sig = wav.read(full_file)
			# open_wav =wave.open(full_file, 'r')
			# rate = open_wav.getframerate()#*DOWNSAMPLE
			# frames = open_wav.getnframes()

			# sig = np.array(open_wav.readframes(frames))
			# print sig
			# open_wav.close()
			for i, split in enumerate(self.split_episodes(sig, rate)):
				mfcc_features = python_speech_features.mfcc(split, rate)

				with open(join(self.mfcc_dir, pitch_audio_fn.split('.')[0] + ".%i" %(i)), 'w')	as output_fn:
					np.savetxt(output_fn, mfcc_features, delimiter= ',')
					print ("Extracted MFCC features for %s, split %i, into: %s") %(pitch_audio_fn, i, output_fn.name)

def write_MFCCs():
	if exists(MFCC_DIR): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST! NB
		print ("Deleting previous mfcc")
		shutil.rmtree(MFCC_DIR)
	os.makedirs(MFCC_DIR)

	if exists(FINAL_LABEL_FILE): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST! NB
		print ('Deleting previous labels')
		os.remove(FINAL_LABEL_FILE)

	labels = defaultdict()
	for season in SEASONS:
		try:
			label_file = "%sseason%i-labelled.p" % (AUDIO_SCRAPING_DIR, season)
			with open(label_file) as of:
				to_add = pickle.loads(of.read())
			labels.update(to_add)
			MFCC_Extractor(season).write_features()
			
		except OSError:
			print ('\n-------\nERROR: Season %i not found!\n--------\n') %(season)

def consolidate_labels():
	if exists(FINAL_LABEL_FILE): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST! NB
		print 'Deleting previous labels'
		os.remove(FINAL_LABEL_FILE)

	with open(FINAL_LABEL_FILE, 'w') as f:
		pickle.dump(labels, f)
		print ("\n----\nLabels compiled into %s\n----\n") %(f) 

if __name__ == '__main__':
	write_MFCCs()
	consolidate_labels()

