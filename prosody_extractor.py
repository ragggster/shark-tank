import os
from os.path import join
from praatio import pitch_and_intensity
import numpy as np

''' 
RELEVANT REFERENCES
-------------------
https://github.com/timmahrt/praatIO/blob/master/praatio/pitch_and_intensity.py
'''

praatEXE = '/Applications/Praat.app/Contents/MacOS/Praat' #change this to your Praat location
currPath = os.getcwd()
outputFile = join(currPath, 'intensityOutput.txt')
wavFile = join(currPath, 's8-e1-p1.wav')

def extract_intensity(wavFile):
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
	return features

def extract_pitch(wavFile):
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
	return features



blah = extract_pitch(wavFile)
print(blah)
