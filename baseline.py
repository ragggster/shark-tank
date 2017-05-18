
import python_speech_features
import tensorflow as tf 
import numpy as np
import scipy.io.wavfile as wav
import os
from os import listdir
from os.path import isfile, join, exists
import wave
from scikits.audiolab import Sndfile
from scikits import audiolab
from unpickle import unpickle


SEASON = 8
DOWNSAMPLE = 1000
EXTRACT = True

DATA_DIR = './audio-scraping/season%s-pitches' %(SEASON)
MFCC_DIR = './data/mfcc' 
META_FILE = './audio-scraping/season%s-pitches-metadata.p' % (SEASON)
LABELS_DIR = './data/labels/'

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if (isfile(join(directory, f)) and f[0] != '.')]

def get_label(line):
	potential_labels = line.split(' ')[0] #ALTER THE INDEX TO CHANGE WHAT YOU ARE PREDICTING!!!!

class MFCC_Extractor():
	def __init__(self, input_dir):
		self.input_fns = get_files_in_dir(input_dir)

	def write_features(self, meta_fn, output_dir):
		# Write features as csv WITH headers so we can pick which to classify later
		# load data
		if not exists(output_dir):
			os.makedirs(output_dir)

		for pitch_audio_fn in self.input_fns:
			
			full_file = join(DATA_DIR, pitch_audio_fn)
			open_wav = Sndfile(full_file)
			rate = open_wav.samplerate*DOWNSAMPLE
			
			sig = open_wav.read_frames(open_wav.nframes)

			# open_wav  = wave.open(full_file, 'r')
			# rate = open_wav.getframerate()
			# sig = open_wav.readframes(open_wav.getnframes())
			#(rate,sig) = wav.read(join(DATA_DIR, pitch_audio_fn))
			mfcc_features = python_speech_features.mfcc(sig, rate)
			print(np.prod(mfcc_features.shape))
			with open(join(output_dir, pitch_audio_fn.split('.')[0]), 'w')	as output_fn:
				np.savetxt(output_fn, mfcc_features, delimiter= ',')

			print "Extracted MFCC features for %s into a csv" %(pitch_audio_fn)

class Baseline():
	def __init__(self):
		self.X_placeholder = tf.placeholder(tf.float32)



	def run_baseline(self, features_dir, labels_dir):
		labels = get_files_in_dir(labels_dir)
		#features = get_files_in_dir(features_dir)
		X, y = [], []
		for label_fn in labels:
			with open(join(labels_dir, label_fn)) as lfn:
				y.append(get_label(lfn.readline()))
			with open(join(features_dir, label_fn.split('.')[0] )) as ffn:
				data = np.loadtxt(ffn, delimiter=',')
				X.append(data.flatten())








if __name__ == '__main__':
	
	if(EXTRACT):
		MFCC_Extractor(DATA_DIR).write_features(META_FILE, MFCC_DIR)

	baseline = Baseline()
	baseline.run_baseline(MFCC_DIR, LABELS_DIR)