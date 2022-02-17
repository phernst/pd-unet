import os
import glob
import nibabel as nib
import numpy as np
import torchcomplex.nn.functional as cF
import torch
import pandas as pd
import random
import scipy.io as sio
from tqdm import tqdm
    
random.seed(13)
np.random.seed(13)

def interpWithTorchComplex(data, size, mode="sinc"):
      data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
      if mode == "sinc":
            data = cF.interpolate(data, size=size, mode=mode)
      else:
            data = cF.interpolate(data+1j, size=size, mode=mode).real
      return data.numpy().squeeze()

def strat_samp(df, percent):
    t2 = percent*df.sum()['T2SPIR']
    t11 = percent*df.sum()['InPhase_01']
    t12 = percent*df.sum()['InPhase_02']

    new_df = []
    while len(new_df) < len(df):
        df_row = df.sample()
        df = df.drop(df_row.Sub)
        if len(new_df) > 0:
            new_df = pd.concat([new_df, df_row])
        else:
            new_df = df_row
        t2_ = percent*new_df.sum()['T2SPIR']
        t11_ = percent*new_df.sum()['InPhase_01']
        t12_ = percent*new_df.sum()['InPhase_02']
        if t2_ >= t2 and t11_ >= t11 and t12_ >= t12:
            return new_df, df
    return new_df, df

source_NIIs = list(glob.glob(r'Data/hrCHAOS/**/*.nii', recursive=True))
destin_root = r"Data/CHAOS/FullyMATs/Normed"

subs = {}
for source_NII in tqdm(source_NIIs):
    shp = nib.load(source_NII).shape[2]
    s = source_NII.split(os.path.sep)[-3]

    if s not in subs:
        subs[s] = {"T2SPIR": 0, "InPhase_01": 0, "InPhase_02": 0}

    if "T2SPIR" in source_NII:
        subs[s]['T2SPIR'] += shp
    elif "InPhase_01" in source_NII:
        subs[s]['InPhase_01'] += shp
    else:
        subs[s]['InPhase_02'] += shp
df = pd.DataFrame.from_dict(subs).T
df['Sub'] = df.index

df_train, df_remain = strat_samp(df, percent=0.6)
df_test, df_val = strat_samp(df_remain, percent=0.6)

n_train = 0
n_test = 0
n_val = 0
random.shuffle(source_NIIs)
for source_NII in tqdm(source_NIIs):
    submame = os.path.basename(source_NII).split('.')[0]
    sID = source_NII.split(os.path.sep)[-3]
    
    vol = np.array(nib.load(source_NII).get_fdata()).astype(np.float32)
    if vol.shape[0] != 256 or vol.shape[1] != 256:
        vol = interpWithTorchComplex(vol, size=(256,256,vol.shape[2]))
    vol = vol / vol.max()

    for sl in range(vol.shape[2]):
        mat = {"subjectName": submame, "fileName": submame+"_sl"+str(sl).zfill(2), "fully": np.expand_dims(vol[...,sl], axis=0)}
        if sID in df_train.Sub:
            p = os.path.join(destin_root, "train", str(n_train)+".mat")
            n_train += 1
        elif sID in df_test.Sub:
            p = os.path.join(destin_root, "test", str(n_test)+".mat")
            n_test += 1
        else:
            p = os.path.join(destin_root, "val", str(n_val)+".mat")
            n_val += 1
        sio.savemat(p, mat)