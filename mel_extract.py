"""
script credit to https://github.com/OckhamsRazor/HSP_CNN
"""

import argparse
import glob
import multiprocessing
from functools import partial
from math import ceil
from os import mkdir, path
from shutil import rmtree
import librosa
import numpy as np
import warnings

# librosa.load will always raise a warning, ignore it
warnings.filterwarnings("ignore")

class ParamsR:

    audio_len = 30
    srt = 22050 # default sample rate
    wsz = 4096 # samples per frame 
    mels = 128
    hop_length = wsz // 2 # determines the overlap between windows
    feat_len = int(ceil(srt * audio_len / float(hop_length)))

params = ParamsR


def _feat_extract(fn, out_p):
    
    # get song id 
    sid = fn.split('/')[-1].split('.')[0]

    # Extract audio timeseries    
    y, sr = librosa.load(fn, sr=params.srt)
    song_len = len(y) / float(params.srt)

    if song_len < params.audio_len/2:
        print(sid)
        
    # Compute a melody-scaled spectrogram, shape = (n_mels, t)
    feat = librosa.feature.melspectrogram(
        y=y, sr=params.srt, n_fft=params.wsz,
        hop_length=params.hop_length, n_mels=params.mels)
    if song_len == params.audio_len:
        ret = feat
    elif song_len > params.audio_len: # select the middle 30s portion 
        start = feat.shape[1] // 2 - params.feat_len // 2
        end = feat.shape[1] // 2 + params.feat_len // 2
        if params.feat_len % 2 == 1:
            end += 1
        ret = feat[:, start:end]
    else: # padd it 
        ret = np.zeros((params.mels, params.feat_len))
        ret[:, :feat.shape[1]] = feat.copy()
    np.save(
        path.join(out_p, sid), ret.reshape(1, params.mels, params.feat_len))
    
    return 0  # success


def main():
    parser = argparse.ArgumentParser(
        description='Song Feat. Extraction')
    parser.add_argument(
        "-i", "--song-dir", help="input song dir", required=True)
    parser.add_argument(
        "-o", "--feat-dir", help="output feat dir", required=True)
    args = parser.parse_args()

    song_dir = args.song_dir
    feat_dir = args.feat_dir
    if path.exists(feat_dir):
        rmtree(feat_dir)
    mkdir(feat_dir)

    songs = glob.glob('{}/*mp3'.format(song_dir))
    pool = multiprocessing.Pool(28)
    excpt = pool.map(partial(_feat_extract, out_p=feat_dir), songs[:])
    pool.close()
    pool.join()
    #for fn in songs: 
    #    _feat_extract(fn, out_p=feat_dir)


if __name__ == "__main__":
    main()

