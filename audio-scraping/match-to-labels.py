import string
import csv
import collections

## Change to set the season for which the download will occur
season = 4

## Generate data structure from .csv file with labels
## Format: array, where index corresponds to season (nothing at 0)
## Within each season, we have a dict from pitch title to a dict of other attributes
label_file = 'season%s.csv' % season
with open(label_file) as csvfile:
    reader = csv.DictReader(csvfile)
    episodes = []
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

        if episodes[episode] is None:

        print (episode_num, pitch_name, label, label_code, deal_amt, deal_equity, deal_val)

## Iterate through all .wav files, map each to a label using the labels_map
## created above
# listing_file = 'season%s-pitches-listing.txt' % season
# lines = open(listing_file, 'r')
# for line in lines:
#     print line
#     elems = line.split('-')
#     season_num, ep_num, pitch_num = elems[0], elems[1], elems[2]
#     season_num = int(''.join(e for e in season_num if e.isdigit()))
#     ep_num = int(''.join(e for e in ep_num if e.isdigit()))
#     pitch_num = int(''.join(e for e in pitch_num if e.isdigit()))
#     if season_num == season:
#         print "At least this is the right season!"
#     pitch_title = "".join(elems[3:])#.join
#     pitch_title = pitch_title.replace('.wav', '')
#
#     print season_num, ep_num, pitch_num
#     print pitch_title
