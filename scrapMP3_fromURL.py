import requests, os, pickle, sys
from bs4 import BeautifulSoup
from util import read_pickle

class Request:
    
    def __init__(self):
        pass

    def request(self) -> requests.Session:
        request_session = requests.Session()
        return request_session
        
def _download_mp3_from_url(requestSession, songID:str, url:str, currPath:str):

    song_dir = os.path.join(currPath, 'preview_mp3') # download to where
    if not os.path.isdir(song_dir): # check dir 
        os.mkdir(song_dir)
    song_name = song_dir + '/' + songID + '.mp3' # song name 
    
    if os.path.isfile(song_name):
        print(f"{song_name} already exists")
        return 
    else: 
        try: 
            print(f"Downloading {url}")
            this_song = requestSession.get(url, stream = True).content # song 
            with open (song_name, 'wb') as f: # save song
                f.write(this_song)

        except: 
            print("Unexpected error:", sys.exc_info()[0])
            raise


def main():

    currPath = os.path.dirname(os.path.realpath(__file__))

    # read in URL pickle file 
    song_info = read_pickle(os.path.join(currPath, 'data', 'songs_genre-mandopop.pkl'))
    preview_urls = song_info[['song_id', 'preview_url']]

    s = Request()
    session = s.request()
    for id, url in zip(preview_urls['song_id'], preview_urls['preview_url']):
        if not url: 
            continue
        else: 
            _download_mp3_from_url(session, id, url, currPath)

    return 0


if __name__ == "__main__":
    
    main()
    sys.exit(0)




