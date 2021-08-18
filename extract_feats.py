import audioread
import librosa
import numpy as np
import os
import shutil
import warnings
import glob
import multiprocessing
from functools import partial

warnings.filterwarnings("ignore")

def extract_melspec(in_fp, win_size):
    in_file = './data/preview_mp3/' + in_fp
    sig, sr = librosa.core.load(in_file, sr=16000)
    feat = librosa.feature.melspectrogram(sig, sr=16000,
                                          n_fft=win_size,
                                          hop_length=512,
                                          n_mels=128).T
    feat = np.log(1+10000*feat)
    
    save_path = 'tag_feature/jy_feat/out'+str(win_size)+'/'
    f = in_fp.split('.')[0]
    np.save(save_path+f+'.npy',feat)
    return 0


if __name__ == '__main__':

    win_sizes = [512,1024,2048,4096,8192,16384]
    
    #remove this in future
    if os.path.exists('tag_feature/jy_feat'):
        shutil.rmtree('tag_feature/jy_feat')
    os.mkdir('tag_feature/jy_feat')

    for win_size in win_sizes:
        path = "./data/preview_mp3/"
        save_path = 'tag_feature/jy_feat/out'+str(win_size)+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        songs = os.listdir(path)
        pool = multiprocessing.Pool(28)
        excpt = pool.map(partial(extract_melspec, win_size = win_size), songs[:])
        pool.close()
        pool.join()
        
    