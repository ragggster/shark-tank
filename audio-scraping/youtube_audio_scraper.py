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
from scikits.audiolab import Format



CSV = "season8-pitches.csv" ## Change link to change season

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

def make_savepath(title, artist, savedir=savedir):
    file_name = ("%s--%s.%s" % (title, artist, options['audioformat']))
    #file_name = '_'.join(file_name.split(' ')) #replaces spaces with underscores
    return os.path.join(savedir, file_name)


downloads = {}

with ydl:

    # read in videos CSV with pandas
    df = pd.read_csv(CSV, sep=";", skipinitialspace=True)
    df.Link = df.Link.map(str.strip)  # strip space from URLs

    # for each row, download
    for _, row in df.iterrows():
        print "Downloading: %s from %s..." % (row.Title, row.Link)

        # download location, check for progress
        savepath = make_savepath(row.Title, row.Artist)
        try:
            os.stat(savepath)
            print "%s already downloaded, continuing..." % savepath
            continue

        except OSError:
            # download video
            try:
                r = ydl.extract_info(row.Link, download=True)
                print(r['id']+ '.' + options['audioformat'] )
                os.rename(r['id']+ '.' + options['audioformat'] , savepath)
                # print r 
                ### Go through the trouble of creating an extra dict here bc otherwise we'd be saving a ton of extraneous info like HTML header stuff and other useless things
                d = defaultdict()
                d['title'] = r['title']
                d['uploader'] = r['uploader']
                d['view_count'] = r['view_count']
                d['like_count'] = r['like_count']
                d['dislike_count'] = r['dislike_count']

                downloads[r['title']] = d

                #print "%s was uploaded by '%s' and has %d views, %d likes, and %d dislikes" % (r['title'], r['uploader'], r['view_count'], r['like_count'], r['dislike_count'])
                #print "Downloaded and converted %s successfully!" % savepath

            except Exception as e:
                print "Can't download audio! %s\n" % traceback.format_exc()

filename = os.path.splitext(CSV)[0] + "-metadata.p"
pkl.dump(downloads, open(filename, "wb"))


