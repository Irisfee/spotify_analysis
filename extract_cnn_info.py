from pathlib import Path
import numpy as np
import pandas as pd
from util import read_pickle, save_csv, save_pickle

base_dir = Path('.')
data_dir = base_dir.joinpath('data')
mel_dir = base_dir.joinpath('mel_features')
test_dir = base_dir.joinpath('test_data')

# save song id
song_id =  np.array([x.stem for x in mel_dir.iterdir()])
np.random.shuffle(song_id)
np.save(test_dir.joinpath('id.npy'), song_id)

mels = []
for song in song_id:
    mel = np.load(mel_dir.joinpath(f'{song}.npy'))
    mels.append(mel)
mels = np.concatenate(mels)
np.save(test_dir.joinpath('feat.npy'), mels)

song_id_sorted = np.load(test_dir.joinpath('id.npy'))

song_url_pop_unique = pd.read_csv(data_dir.joinpath("songs_url_pop-mandopop.tsv"), sep='\t')
pop_rank = []
for song in song_id_sorted:
    d = song_url_pop_unique[song_url_pop_unique['song_id']==song]
    pop_rank.append(d.popularity)
pop_rank = np.concatenate(pop_rank)
np.save(test_dir.joinpath('pop.npy'), pop_rank)

song_url_pop_unique = pd.read_csv(data_dir.joinpath("songs_url_pop-mandopop.tsv"), sep='\t')
feature = read_pickle(data_dir.joinpath("songs_genre-mandopop.pkl"))
feature_rm = feature.drop(columns=['id', 'name','album_id','release_date']).drop_duplicates()
songs_mandopop = song_url_pop_unique.merge(feature_rm,on=['song_id', 'preview_url', 'popularity'], how = 'inner')
save_pickle(songs_mandopop, data_dir.joinpath('songs_mandopop.pkl'))