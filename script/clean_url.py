import pandas as pd
import numpy as np
import glob, os

# get unique mp3 name 
unique_song_id_df = pd.read_csv('./data/songs_url_pop-mandopop.tsv', sep='\t') 
unique_song_id = unique_song_id_df['song_id']
unique_song_id_mp3 = [song_id + '.mp3' for song_id in unique_song_id]

# get all mp3 name
preview_mp3_dir = './preview_mp3/'
preview_song_id_mp3 = os.listdir(preview_mp3_dir)

# get mp3 id that need to remove 
remove_song_id_mp3 = list(set(preview_song_id_mp3) - set(unique_song_id_mp3))
remove_song_id_dir = [preview_mp3_dir + song_id for song_id in remove_song_id_mp3]

print(f"Currently there are {len(preview_song_id_mp3)} mp3 among which there are \
    {len(unique_song_id_mp3)} unique songs, and thus need to remove {len(remove_song_id_dir)} songs")
print('Starting to remove')
# remove 
for song_dir in remove_song_id_dir:
    if os.path.exists(song_dir):
        os.remove(song_dir)
    else:
        print("The file does not exist")



