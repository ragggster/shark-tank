
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

print (audiolab.available_file_formats())


DATA_DIR = './audio-scraping/season8-pitches'
MFCC_DIR = './data/mfcc' 

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if (isfile(join(directory, f)) and f[0] != '.')]


class MFCC_Extractor():
	def __init__(self, input_dir):
		self.input_fns = get_files_in_dir(input_dir)


	def write_features(self, output_dir):
		# Write features as csv WITH headers so we can pick which to classify later
		# load data
		if not exists(output_dir):
			os.makedirs(output_dir)


		for pitch_audio_fn in self.input_fns:
			full_file = join(DATA_DIR, pitch_audio_fn)
			open_wav = Sndfile(full_file)
			rate = open_wav.samplerate
			
			sig = open_wav.read_frames(open_wav.nframes)

			# open_wav  = wave.open(full_file, 'r')
			# rate = open_wav.getframerate()
			# sig = open_wav.readframes(open_wav.getnframes())
			#(rate,sig) = wav.read(join(DATA_DIR, pitch_audio_fn))
			mfcc_features = python_speech_features.mfcc(sig, rate)
			
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