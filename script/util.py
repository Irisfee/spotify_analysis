# -*- coding: utf-8 -*-
"""
Useful functions for spotify analysis project.

author: Yufei Zhao
date: 2021.7.27
"""
import pandas as pd
from pathlib import Path
import pickle

def save_csv(data, fid):
    data.to_csv(fid, sep='\t', float_format='%.5f', na_rep='n/a', index=False)
    
def save_pickle(data, fid):
    open_file = open(fid, "wb")
    pickle.dump(data, open_file)
    open_file.close()
    
def read_pickle(fid):
    open_file = open(fid, "rb")
    data = pickle.load(open_file)
    open_file.close()
    return data


