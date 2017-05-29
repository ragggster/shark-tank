'''
Make sure you pip install youtube-dl prior to running this code
'''

import youtube_dl
import pandas as pd
import os
import traceback
import pickle as pkl
from collections import defaultdict
from unpickle import unpickle
#from scikits.audiolab import Format
from collections import Counter
import re

CSV = "season5-pitches.csv" ## Change link to change season

# create directory
savedir = os.path.splitext(CSV)[0]
if not os.path.exists(savedir):
    os.makedirs(savedir)

# create YouTube downloader
options = {
    'format': 'bestaudio/best', # choice of quality
    'extractaudio' : True,      # only keep the audio
    'audioformat' : "wav",      # convert to mp3
    'outtmpl': '%(id)s.%(ext)s',        # name the file the ID of the video
    'noplaylist' : True,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }]
    }       # only download single song, not playlist


ydl = youtube_dl.YoutubeDL(options)

def make_savepath(encoded_name, savedir=savedir):
    file_name = ("%s.%s" % (encoded_name, options['audioformat']))
    #file_name = '_'.join(file_name.split(' ')) #replaces spaces with underscores
    return os.path.join(savedir, file_name)

downloads = {}

with ydl:

    # read in videos CSV with pandas
    df = pd.read_csv(CSV, sep=";", skipinitialspace=True)
    df.Link = df.Link.map(str.strip)  # strip space from URLs

    name_counter = Counter() #Used to count which pitch we are on
    season_regex = re.compile('[Ss]eason[ \t]*(?P<season>\d+)')
    episode_regex = re.compile('[Ee]pisode[ \t]*(?P<episode>\d+)')

    # for each row, download
    for _, row in df.iterrows():
        print "Downloading: %s from %s..." % (row.Title, row.Link)

        season = season_regex.search(row.Title).group('season')
        episode = episode_regex.search(row.Title).group('episode')
        enc_name = 's' + str(season) + '-e' + str(episode)
        name_counter.update([enc_name])
        enc_name = enc_name + '-p' + str(name_counter[enc_name]) + '-'

        try:
            # re-download video irrespective of whether it's already around, we need to download to extract relevant info
            r = ydl.extract_info(row.Link, download=True)

            # get the title format right
            title = r['title']
            idx = title.find('(')
            title = title[:idx]
            title = title.replace('pitch', '')
            title = title.replace('shark tank', '')
            title = title.replace(' ', '-').lower()
            title = ''.join(e for e in title if e.isalnum() or e == '-')
            title = title.replace('shark-tank', '')
            title = title[:-1]
            enc_name += title
            enc_name = enc_name.replace('--', '-')
            enc_name = enc_name.replace('-pitch', '')

            # download location, check for progress
            savepath = make_savepath(enc_name)
            try:
                os.stat(savepath)
                print "%s already downloaded, continuing..." % savepath
                continue

            except OSError:

                os.rename(r['id']+ '.' + options['audioformat'] , savepath)
                # Go through the trouble of creating an extra dict here bc
                # otherwise we'd be saving a ton of extraneous info like HTML header stuff and other useless things
                d = defaultdict()
                d['title'] = r['title']
                d['uploader'] = r['uploader']
                d['view_count'] = r['view_count']
                d['like_count'] = r['like_count']
                d['dislike_count'] = r['dislike_count']


                downloads[enc_name] = d

                #print "%s was uploaded by '%s' and has %d views, %d likes, and %d dislikes" % (r['title'], r['uploader'], r['view_count'], r['like_count'], r['dislike_count'])
                #print "Downloaded and converted %s successfully!" % savepath

        except Exception as e:
            print "Can't download audio! %s\n" % traceback.format_exc()

filename = os.path.splitext(CSV)[0] + "-metadata.p"
pkl.dump(downloads, open(filename, "wb"))
