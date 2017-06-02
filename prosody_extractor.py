import os
from os.path import isfile, join, exists
from os import listdir
from praatio import pitch_and_intensity
import numpy as np
import python_speech_features
import scipy.io.wavfile as wav
import shutil

''' 
RELEVANT REFERENCES
-------------------
https://github.com/timmahrt/praatIO/blob/master/praatio/pitch_and_intensity.py
'''

DATA_DIR = './data'
PROS_DIR = DATA_DIR + '/prosody'
F0_DIR = join(PROS_DIR, "f0")
INTENSITY_DIR = join(PROS_DIR, "intensity")
AUDIO_SCRAPING_DIR = "./audio-scraping/"

SEASONS = [1, 2, 3, 4, 5, 6, 7, 8]
SPLIT_TIME = 10 #in seconds
MIN_SIZE = 250000 #in number of samples, 250000 is about 6 seconds

praatEXE = '/Applications/Praat.app/Contents/MacOS/Praat' #change this to your Praat location
CURR_PATH = os.getcwd() #Praat scripts need absolute path :(
outputFile = join(CURR_PATH, 'prosodicOutput.txt')

def get_files_in_dir(directory):
	# http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
	return [f for f in listdir(directory) if (isfile(join(directory, f)) and f[0] != '.')]

class Prosody_Extractor():
	def __init__(self, season, split_time = SPLIT_TIME):
		self.data_dir = '%sseason%s-pitches' %(AUDIO_SCRAPING_DIR, season)
		self.pros_dir = PROS_DIR
		self.f0_dir = F0_DIR
		self.intensity_dir = INTENSITY_DIR
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
			print ('split time: %f seconds (%i samples)' %(1.0*l/rate, l))
			yield signal[seg_start:seg_end]

	# extracting f0 (aka pitch) from the wav file segment
	# note that the outputFile generated here is useless to us but the function needs it so just ignore it
	def extract_f0_from_segment(self, wav_file):
		pitchList = pitch_and_intensity.extractPitch(wav_file, outputFile, praatEXE, 35, 600)
		pitchList = np.array(pitchList)
		pitches = pitchList[:,1]
		return pitches

	# extracting intensity from the wav file segment
	def extract_intensity_from_segment(self, wav_file):
		intenseList = pitch_and_intensity.extractIntensity(wav_file, outputFile, praatEXE, 100)
		intenseList = np.array(intenseList)
		intensities = intenseList[:,1] #remove the time col
		return intensities

	def write_features(self):
		# create a temp dir to store the split wav files
		temp_dir = join(CURR_PATH, self.data_dir + "-temp")
		if exists(temp_dir): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST!
			print ("Deleting previous temp directory")
			shutil.rmtree(temp_dir)
		os.makedirs(temp_dir)

		for pitch_audio_fn in self.input_fns:
			full_file = join(self.data_dir, pitch_audio_fn)
			rate, sig = wav.read(full_file)
			for i, split in enumerate(self.split_episodes(sig, rate)):
				# write out the 10-sec wav files
				segment_file = pitch_audio_fn.split('.')[0] + "-%i.wav" %(i)
				segment_file = join(temp_dir, segment_file)
				wav.write(segment_file, rate, split)

				# extract f0 features
				f0_features = self.extract_f0_from_segment(segment_file)
				# extract intensity features
				intensity_features = self.extract_intensity_from_segment(segment_file)

				# for a file, each line corresponds to the f0 extracted for that time frame
				with open(join(self.f0_dir, pitch_audio_fn.split('.')[0] + ".%ip" %(i)), 'w') as output_file:
					np.savetxt(output_file, f0_features, delimiter= ',')
					print ("Extracted F0 features for %s, split %i, into: %s" %(pitch_audio_fn, i, output_file.name))

				# for a file, each line corresponds to the intensity extracted for that time frame
				with open(join(self.intensity_dir, pitch_audio_fn.split('.')[0] + ".%ii" %(i)), 'w') as output2_file:
					np.savetxt(output2_file, intensity_features, delimiter= ',')
					print ("Extracted intensity features for %s, split %i, into: %s" %(pitch_audio_fn, i, output2_file.name))

				# delete the segmented wav file after extracting the relevant features
				os.remove(segment_file)

		# remove the useless output file
		if (isfile("prosodicOutput.txt")):
			os.remove("prosodicOutput.txt")


# Handle creating/deleting of the directories for prosodic features
def write_prosody():
	if not exists(PROS_DIR): 
		os.makedirs(PROS_DIR)
	# write pitch/f0 files
	if exists(F0_DIR): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST!
		print ("Deleting previous f0 directory")
		shutil.rmtree(F0_DIR)
	os.makedirs(F0_DIR)
	# write intensity files
	if exists(INTENSITY_DIR): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST!
		print ("Deleting previous intensity directory")
		shutil.rmtree(INTENSITY_DIR)
	os.makedirs(INTENSITY_DIR)


'''def extract_intensity(wavFile):
	# 100 is the minimum pitch
	intenseList = pitch_and_intensity.extractIntensity(wavFile, outputFile, praatEXE, 100)
	intenseList = np.array(intenseList)
	intensities = intenseList[:,1] #remove the time col

	# compute mean, median, var, max, min, range
	i_mean = np.mean(intensities)
	i_median = np.median(intensities)
	i_var = np.var(intensities)
	#i_min = np.amin(intensities)
	i_max = np.amax(intensities)
	i_min = np.percentile(intensities, 1) #excluding outliers
	i_range = i_max - i_min

	features = np.array([i_mean, i_median, i_var, i_min, i_max, i_range])
	return features'''

'''def extract_pitch(wavFile):
	pitchList = pitch_and_intensity.extractPitch(wavFile, outputFile, praatEXE, 35, 600)
	pitchList = np.array(pitchList)
	pitches = pitchList[:,1]
	# compute mean, median, var, max, min, range
	p_mean = np.mean(pitches)
	p_median = np.median(pitches)
	p_var = np.var(pitches)
	p_min = np.percentile(pitches, 5)
	p_max = np.percentile(pitches, 95) #excluding outliers
	p_range = p_max - p_min

	features = np.array([p_mean, p_median, p_var, p_min, p_max, p_range])
	return features'''

if __name__ == '__main__':
	write_prosody()
	for season in SEASONS:
		try:
			Prosody_Extractor(season).write_features()	
		except OSError:
			print ('\n-------\nERROR: Season %i not found!\n--------\n') %(season)
