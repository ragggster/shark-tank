
import python_speech_features
import tensorflow as tf 
import numpy as np
import scipy.io.wavfile as wav
import os
from os import listdir
from os.path import isfile, join, exists


DATA_DIR = './audio-scraping/season8-pitches'
MFCC_DIR = './data/mfcc' 

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if isfile(join(directory, f))]


class MFCC_Extractor():
	def __init__(self, input_dir):
		self.input_fns = get_files_in_dir(input_dir)


	def write_features(self, output_dir):
		# Write features as csv WITH headers so we can pick which to classify later
		# load data
		if not exists(output_dir):
			os.makedirs(output_dir)


		for pitch_audio_fn in self.input_fns:
			print(pitch_audio_fn)
			(rate,sig) = wav.read(pitch_audio_fn)
			mfcc_features = python_speech_feature.mfcc(sig, rate)
			print(pitch_audio_fn)
			continue
			audio_fn_base = pitch_audio_fn.split('/')[-1]




class Baseline():
	def __init__():
		pass



	def run_baseline(features_fns):
		filename_queue = tf.train.string_input_producer(features_fns)
		

		
			








if __name__ == '__main__':
	MFCC_Extractor(DATA_DIR).write_features(MFCC_DIR)

	baseline = Baseline()
	baseline.run_baseline(MFCC_DIR)