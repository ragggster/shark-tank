from pydub import AudioSegment
import os
import csv
from os.path import join

''' 
NOTE
---------
the current test csv i'm using is season7-pitchinfo.csv
this is fake data!!
the current test csv data is formatted as:
# s8-e1-p1.wav, 0:10|0:40|cookie-kahuna, 1:00|1:10|bobs-burgers ...
'''

# this function extracts start up pitches from the full episodes downloaded

class Pitch_Extractor():

	def __init__(self, season):
		new_pitches_dir = './season%s-pitches' %(season)
		self.season = season
		self.data_dir = new_pitches_dir
		self.pitch_times_file = 'season%s-pitchinfo.csv' %(season) #i.e. season7-pitchinfo.csv
		if not os.path.exists(new_pitches_dir):
			os.makedirs(new_pitches_dir)

	'''
	Helper function: takes a time string formatted mm:ss and converts it to milliseconds
	This is because pydub operates in milliseconds
	'''
	def get_milliseconds(self, time_str):
		m, s = time_str.split(':')
		seconds = int(m)*60.0 + int(s)
		return seconds*1000

	'''
	For each source wav file in the .csv, we clip out all the specified pitches using start and end times
	Generated pitch files are written out to a directory for the season, i.e. season7-pitches
	Current example of name of generate pitch file: s7-cookie-kahuna.wav
	'''
	def extract_pitches(self):
		with open(self.pitch_times_file, 'rU') as csvfile:
			readCSV = csv.reader(csvfile, dialect=csv.excel_tab, delimiter=',')
			for row in readCSV:
				source_file = row[0]
				full_path = join('./season6-source-pitches', source_file)
				pitch_source = AudioSegment.from_wav(full_path)
				source_len = pitch_source.duration_seconds*1000 #total length of source file in ms
				for i in range(1, len(row)):
					pitch_info = row[i].split('|')
					if (len(pitch_info) <= 1): continue
					print(pitch_info)
					start_time = pitch_info[0]
					end_time = pitch_info[1]
					pitch_name = pitch_info[2]
					start_time_ms = self.get_milliseconds(start_time)
					end_time_ms = self.get_milliseconds(end_time)
					assert(start_time_ms <= source_len)
					assert(end_time_ms <=source_len)
					print(str(start_time_ms) + " " + str(end_time_ms))
					pitch_clip = pitch_source[start_time_ms:end_time_ms]
					#pitch_filepath = self.data_dir + "/" + "s%s-%s.wav" %(self.season, pitch_name)
					source_name = os.path.splitext(source_file)[0] #remove extension
					pitch_filepath = self.data_dir + "/" + "%s-%s.wav" %(source_name, pitch_name)
					pitch_clip.export(pitch_filepath, format='wav')

pe = Pitch_Extractor("6") #example
pe.extract_pitches()
