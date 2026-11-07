import sys
from scipy.io import loadmat, whosmat
for p in sys.argv[1:]:
    print(p, whosmat(p))
    m = loadmat(p)
    for k, v in m.items():
        if k.startswith('__'): print(k, v); continue
        print(k, type(v), getattr(v, 'shape', None), getattr(v, 'dtype', None))
        if k == 'gt':
            for i, x in enumerate(v.ravel()):
                print(i+1, x.tolist())
        else:
            import numpy as np
            a = np.asarray(v).ravel()
            print('  len', a.size, 'sum', a.sum(), 'uniq', np.unique(a)[:5])
