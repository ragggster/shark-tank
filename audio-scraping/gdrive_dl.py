import requests
import os
from os.path import join

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    #file_id = 'TAKE ID FROM SHAREABLE LINK'
    season = 1
    new_pitches_dir = './season%s-pitches' %(season)
    if not os.path.exists(new_pitches_dir):
            os.makedirs(new_pitches_dir)
    file_id = '0B0Brggz-K4BMd2xSQm9KT2RtR1U'
    destination = join(new_pitches_dir, 's1-e9-virtusphere.wav')
    download_file_from_google_drive(file_id, destination)