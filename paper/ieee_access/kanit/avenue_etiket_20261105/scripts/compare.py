import pickle, sys, itertools
import numpy as np
S = sys.argv[1]
d = pickle.load(open(f'{S}/all.pkl', 'rb'))
ids = [f'{i:02d}' for i in range(1, 22)]
L = {k: len(d['ours'][k]) for k in ids}
vpc = {}
for k in ids:
    a = np.zeros(L[k], np.uint8)
    for s, e in d['vpc_iv'][k]:
        a[s-1:e] = 1
    vpc[k] = a
src = dict(ours=d['ours'], cuhk=d['cuhk'], vpc=vpc, acci=d['acci'], aed=d['aed'])
def iv(mask):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j+1]: j += 1
            out.append([i+1, j+1]); i = j + 1
        else: i += 1
    return out
print('totals:', {s: int(sum(v[k].sum() for k in ids)) for s, v in src.items()}, 'frames', sum(L.values()))
print('vid', *[f'{s:>5}' for s in src])
for k in ids:
    print(k, *[f'{int(v[k].sum()):>5}' for v in src.values()])
for a, b in itertools.combinations(src, 2):
    nd = sum(int((src[a][k] != src[b][k]).sum()) for k in ids)
    print(f'{a} vs {b}: {nd} differing frames')
print()
for s in ['cuhk', 'vpc', 'acci', 'aed']:
    print('intervals', s)
    for k in ids:
        print(' ', k, iv(src[s][k]))
