import string
import csv
from collections import defaultdict
# import pickle
# import marshal, types
import dill as pickle

## Change to set the season for which the download will occur
season = 4

## Generate data structure from .csv file with labels
## Format: array, where index corresponds to season (nothing at 0)
## Within each season, we have a dict from pitch title to a dict of other attributes
label_file = 'season%s.csv' % season
with open(label_file) as csvfile:
    reader = csv.DictReader(csvfile)
    episodes = [None] * 100
    for row in reader:
        season_num = int(row['Season'])
        episode_num = int(row['No. in series'])
        pitch_name = row['Company']
        label = row['Deal']
        if label == 'Yes':
            label_code = 1
        else:
            label_code = 0
        gender = row['Entrepreneur Gender']
        if (row['Amount']):
            deal_amt = float(row['Amount'])
        else:
            deal_amt = 0.0
        if (row['Equity']):
            deal_equity = float(row['Equity'])
        else:
            deal_equity = 0.0
        if (row['Valuation']):
            deal_val = float(row['Valuation'])
        else:
            deal_val = 0.0
        print "Processing label for episode %s: %s pitch" % (episode_num, pitch_name)

        info = defaultdict(str)
        info['label'] = label
        info['label_code'] = label_code
        info['gender'] = gender
        info['deal_amt'] = deal_amt
        info['deal_equity'] = deal_equity
        info['deal_val'] = deal_val
        if episodes[episode_num] is None:
            d = defaultdict(lambda: defaultdict(str))
            d[pitch_name] = info
            episodes[episode_num] = d
        else:
            d = episodes[episode_num]
            d[pitch_name] = info
            episodes[episode_num] = d
        print (episode_num, pitch_name, label, label_code, deal_amt, deal_equity, deal_val)

## Iterate through all .wav files, map each to a label using the labels_map
## created above
correct_label_mappings = defaultdict(lambda: defaultdict(str))
listing_file = 'season%s-pitches-listing.txt' % season
lines = open(listing_file, 'r')
for line in lines:
    # print line
    elems = line.split('-')
    season_num, ep_num, pitch_num = elems[0], elems[1], elems[2]
    season_num = int(''.join(e for e in season_num if e.isdigit()))
    ep_num = int(''.join(e for e in ep_num if e.isdigit()))
    pitch_num = int(''.join(e for e in pitch_num if e.isdigit()))
    pitch_title = "".join(elems[3:])#.join
    pitch_title = pitch_title.replace('.wav', '')
    line = line.replace('\n', '')
    print "Does %s correspond to: " % line
    temp_pitch_map = [None]*10
    i = 0
    for k, v in episodes[ep_num].iteritems():
        temp_pitch_map[i] = k
        print "%s: %s" % (i, k)
        i += 1
    while(True):
        input_var = raw_input("Enter the number corresponding to the correct pitch: ")
        if input_var.isdigit():
            num = int(input_var)
            if num <= i-1:
                break
        print "Invalid input."
    print "Mapping %s to label info for %s: " % (line, temp_pitch_map[num])
    print episodes[ep_num][temp_pitch_map[num]]
    correct_label_mappings[line] = episodes[ep_num][temp_pitch_map[num]]

## Working on figuring out how to pickle this right now
print correct_label_mappings
pickle_filename = 'season%s-labelled.p'% season
pickle.dump(correct_label_mappings, open(pickle_filename, "wb"))
