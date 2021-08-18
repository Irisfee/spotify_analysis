import theano
import theano.tensor as T
import audioread
import librosa
from lasagne import layers
from math import ceil
import numpy as np
from clip2frame import utils, measure
import network_structure as ns
import os, shutil, warnings, glob, json, ast, warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Extract mel spectrum features at different hop length
def _extract_melspec(in_file, win_size):
    
    """
    In_file is a path pointing to a MP3 file
    output is of the shape (#Frame, #mel)
    """
    
    sig, sr = librosa.core.load(in_file, sr=16000)
    feat = librosa.feature.melspectrogram(sig, sr=16000,
                                          n_fft=win_size,
                                          hop_length=512,
                                          n_mels=128).T
    feat = np.log(1+10000*feat)
    return feat

# padding if sone less than 30s
def _append_zero_row(array, n_total_row):
    r, c = array.shape
    if r >= n_total_row:
        return array
    else:
        temp = np.zeros((n_total_row-r, c))
        return np.vstack([array, temp])
    

def _make_batch_feat(feature_list, length=939):
    
    """
    feature_list is the list of output of _extract_melspec, numpy array. 
    If only one song, need to make it into a list
    
    output file is of the shape (nSong, 1, nFrame, nMel)
    """
    
    feat = []
    a_array = []
    
    for idx, term in enumerate(feature_list):

        a_array = _append_zero_row(term, length)[None, None, :length, :].astype('float32')
        feat.append(a_array)

    feat = np.vstack(feat)
    return feat

def _standardize(batch_feat, scaler=None):
    
    k = batch_feat.shape[-1]
    n = batch_feat.shape[0]
    batch_feat = batch_feat.reshape((-1, k))
    
    scaler = StandardScaler().fit(batch_feat)
    batch_feat_s = scaler.transform(batch_feat)
    
    batch_feat_s = batch_feat_s.reshape((n, 1, -1, k))
    return batch_feat_s

# tag feature 
def predict_curSong(mp3_file):
    # Test settings
    build_func = ns.build_fcn_gaussian_multiscale
    test_measure_type_list = ['mean_auc_y', 'mean_auc_x', 'map_y', 'map_x']
    n_top_tags_te =50  # Number of tags used for testing

    #threshold setting
    tag_tr_fp = 'data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = 'data/data.magnatagatune/tag_list.top50.txt'

    tag_idx_list = utils.get_test_tag_50(tag_tr_fp, tag_te_fp)

    model_dir = 'data/models'
    thres_fp = os.path.join(model_dir, 'threshold.20160309_111546.with_magnatagatune.npy')
    thresholds_raw = np.load(thres_fp)
    thresholds = thresholds_raw[tag_idx_list]

    # Test data directory
    # The complete MagnaATagATune training/testing data can be downloaded from
    # http://mac.citi.sinica.edu.tw/~liu/data/exp_data.MagnaTagATune.188tags.zip
    # After downloading, replace the data_dir with the new directory path
    use_real_data = False

    # data_dir = '../data/HL30/standardize/'
    # data_dir = '/gpfs/projects/hulacon/shared/nsd_results/yufei/spotify_analysis/tag_feature/std/'

    # Files
    param_fp = 'data/models/model.20160309_111546.npz'
    tag_tr_fp = 'data/data.magnatagatune/tag_list.top188.txt'
    tag_te_fp = 'data/data.magnatagatune/tag_list.top{}.txt'.format(n_top_tags_te)

    # Load tag list
    tag_tr_list = utils.read_lines(tag_tr_fp)
    tag_te_list = utils.read_lines(tag_te_fp)

    label_idx_list = [tag_tr_list.index(tag) for tag in tag_te_list]

    X_te_list = []
    # Load data
    print("Loading data...")
    
    for win_size in [512,1024,2048,4096,8192,16384]:
        
        single_feature = _extract_melspec(mp3_file, win_size) # extract feature 
        batch_feature = _make_batch_feat([single_feature]) #add padding
        batch_feature_s = _standardize(batch_feature) # standarize 
        X_te_list.append(batch_feature_s)
        
    #X_te_list.append(np.load(data_dir +'/feat_test_512.npy'))
    #X_te_list.append(np.load(data_dir +'/feat_test_1024.npy'))
    #X_te_list.append(np.load(data_dir +'/feat_test_2048.npy'))
    #X_te_list.append(np.load(data_dir +'/feat_test_4096.npy'))
    #X_te_list.append(np.load(data_dir +'/feat_test_8192.npy'))
    #X_te_list.append(np.load(data_dir +'/feat_test_16384.npy'))

    print(len(X_te_list), len(X_te_list[0]), len(X_te_list[0][0]), len(X_te_list[0][0][0]))

    # Building Network
    print("Building network...")
    num_scales = 6
    network, input_var_list, _, _ = build_func(num_scales)

    # Computing loss
    target_var = T.matrix('targets')
    epsilon = np.float32(1e-6)
    one = np.float32(1)

    output_va_var = layers.get_output(network, deterministic=True)
    output_va_var = T.clip(output_va_var, epsilon, one-epsilon)

    func_pr = theano.function(input_var_list, output_va_var)

    # Load params
    utils.load_model(param_fp, network)

    # Predict
    print('Predicting...')

    pred_list_raw = utils.predict_multiscale(X_te_list, func_pr)
    pred_all_raw = np.vstack(pred_list_raw)

    pred_all = pred_all_raw[:, label_idx_list]
    pred_binary = ((pred_all-thresholds) > 0).astype(int)
    
    return pred_all

# mel feature
def feat_extract(mp3_file):
    
    class ParamsR:

        audio_len = 30
        srt = 22050 # default sample rate
        wsz = 4096 # samples per frame 
        mels = 128
        hop_length = wsz // 2 # determines the overlap between windows
        feat_len = int(ceil(srt * audio_len / float(hop_length)))

    params = ParamsR
    
    # Extract audio timeseries    
    y, sr = librosa.load(mp3_file, sr=params.srt)
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
    
    return ret



    

