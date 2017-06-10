import os
from os import listdir
from os.path import isfile, join, exists
import shutil
import dill as pickle
from collections import defaultdict

DATA_DIR = './data'
AUDIO_SCRAPING_DIR = "./audio-scraping/"
FINAL_LABEL_FILE = DATA_DIR + '/labels.p'
SEASONS = [1, 2, 3, 4, 5, 6, 7, 8]

labels = dict()
for season in SEASONS:
	try:
		label_file = "%sseason%i-labelled-new.p" % (AUDIO_SCRAPING_DIR, season)
		with open(label_file, 'r+') as of:
			to_add = pickle.loads(of.read())
		labels.update(to_add)			
	except OSError:
		print ('\n-------\nERROR: Season %i not found!\n--------\n') %(season)

if exists(FINAL_LABEL_FILE): #DELETES ALL PRE-EXISTING FEATURE DATA FIRST! NB
	print 'Deleting previous labels'
	os.remove(FINAL_LABEL_FILE)


# pickle.dump(correct_label_mappings, open(pickle_filename, "wb"))
pickle.dump(labels, open(FINAL_LABEL_FILE, "wb"))
print ("\n----\nLabels compiled into %s\n----\n") %(FINAL_LABEL_FILE) 