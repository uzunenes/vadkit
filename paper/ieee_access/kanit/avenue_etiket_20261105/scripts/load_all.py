import json, glob, os, pickle, sys
import numpy as np
from scipy.io import loadmat
S = sys.argv[1]
ACCI = '/home/lei/Desktop/msc/ext/Accurate-Interpretable-VAD/data/avenue.mat'
ids = [f'{i:02d}' for i in range(1, 22)]
ours = {k: np.array(v, dtype=np.uint8) for k, v in json.load(open(f'{S}/remote/m3_gt.json')).items()}
# CUHK masks
cuhk, npix, nlab = {}, {}, {}
for i in range(1, 22):
    vol = loadmat(f'{S}/x_gtdemo/ground_truth_demo/testing_label_mask/{i}_label.mat')['volLabel'].ravel()
    c = np.array([int(np.asarray(m).sum()) for m in vol])
    shp = {np.asarray(m).shape for m in vol}
    npix[f'{i:02d}'] = c; cuhk[f'{i:02d}'] = (c > 0).astype(np.uint8); nlab[f'{i:02d}'] = (len(vol), shp)
vm = loadmat(f'{S}/remote/vpc_avenue.mat')['gt'].ravel()
vpc_iv = {ids[i]: np.asarray(vm[i]).reshape(2, -1).T.tolist() for i in range(21)}
am = loadmat(ACCI)
acci = {k: am[k].ravel().astype(np.uint8) for k in ids}
aed = {k: (np.loadtxt(f'{S}/x_aedmae/Avenue_gt/{k}.txt') > 0.5).astype(np.uint8) for k in ids}
aed_raw_uniq = sorted({float(x) for k in ids for x in np.unique(np.loadtxt(f'{S}/x_aedmae/Avenue_gt/{k}.txt'))})
pickle.dump(dict(ours=ours, cuhk=cuhk, npix=npix, nlab=nlab, vpc_iv=vpc_iv, acci=acci, aed=aed, aed_raw_uniq=aed_raw_uniq), open(f'{S}/all.pkl', 'wb'))
for k in ids:
    print(k, 'len ours', len(ours[k]), 'cuhk', nlab[k][0], nlab[k][1], 'acci', len(acci[k]), 'aed', len(aed[k]))
print('aed raw values', aed_raw_uniq)
