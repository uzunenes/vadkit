import pickle, sys, json, hashlib, subprocess, itertools
import numpy as np
S, OUT = sys.argv[1], sys.argv[2]
d = pickle.load(open(f'{S}/all.pkl', 'rb'))
ids = [f'{i:02d}' for i in range(1, 22)]
L = {k: len(d['ours'][k]) for k in ids}
vpc = {}
for k in ids:
    a = np.zeros(L[k], np.uint8)
    for s, e in d['vpc_iv'][k]:
        a[s-1:e] = 1
    vpc[k] = a
src = {'ours_gt_json': d['ours'], 'cuhk_mask_anypixel': d['cuhk'], 'vpc_avenue_mat': vpc,
       'accivad_avenue_mat': d['acci'], 'aedmae_avenue_gt_txt': d['aed']}
def iv(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j+1]: j += 1
            out.append([i+1, j+1]); i = j + 1
        else: i += 1
    return out
C, V, P = d['cuhk'], vpc, d['npix']
per = {}
cat_tot = {}
cuhk_only_area = []
all_pos_area = []
for k in ids:
    n = L[k]
    c_iv, v_iv = iv(C[k]), d['vpc_iv'][k]
    co = iv((C[k] == 1) & (V[k] == 0)); vo = iv((V[k] == 1) & (C[k] == 0))
    cats = []
    for s, e in vo:
        seg_c = C[k][s-1:e]
        # VPC interval containing this segment
        vi = [x for x in v_iv if x[0] <= s and e <= x[1]][0]
        if C[k][vi[0]-1:vi[1]].sum() == 0:
            t = 'vpc_extra_event_no_mask'
        elif s > vi[0] and e < vi[1]:
            t = 'vpc_gap_fill_between_mask_runs'
        else:
            t = 'vpc_other'
        cats.append(dict(frames=[s, e], n=e-s+1, only_in='vpc', type=t))
    for s, e in co:
        run = [x for x in c_iv if x[0] <= s and e <= x[1]][0]
        if V[k][run[0]-1:run[1]].sum() == 0:
            t = 'cuhk_event_dropped_in_vpc'
        elif e < min(x[0] for x in v_iv if x[1] >= run[0] and x[0] <= run[1]):
            t = 'vpc_starts_event_later'
        elif e == n:
            t = 'vpc_ends_event_earlier_at_video_end'
        else:
            t = 'cuhk_other'
        a = P[k][s-1:e]
        cuhk_only_area += a.tolist()
        cats.append(dict(frames=[s, e], n=e-s+1, only_in='cuhk', type=t,
                         mask_pixels_min=int(a.min()), mask_pixels_median=float(np.median(a)), mask_pixels_max=int(a.max())))
    for c in cats:
        cat_tot.setdefault(c['type'], [0, 0]); cat_tot[c['type']][0] += 1; cat_tot[c['type']][1] += c['n']
    all_pos_area += P[k][C[k] == 1].tolist()
    per[k] = dict(n_frames=n,
                  positives={s: int(v[k].sum()) for s, v in src.items()},
                  n_diff_cuhk_vs_vpc=int((C[k] != V[k]).sum()),
                  n_events_vpc_intervals=len(v_iv), n_runs_cuhk=len(c_iv),
                  cuhk_runs=c_iv, vpc_intervals=v_iv,
                  only_cuhk=co, only_vpc=vo, diff_segments=sorted(cats, key=lambda c: c['frames'][0]))
def h(p):
    b = open(p, 'rb').read()
    return dict(md5=hashlib.md5(b).hexdigest(), git_blob_sha1=hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest(), bytes=len(b))
files = {
 'ours_gt_json': (f'{S}/remote/m3_gt.json', 'cctv-aean/prepare_avenue.py -> /workspace/cache/avenue/gt.json (m3)'),
 'cuhk_mask_anypixel': (f'{S}/remote/ground_truth_demo.zip', 'official CUHK ground_truth_demo.zip/testing_label_mask/<i>_label.mat (volLabel), frame=1 iff mask has any nonzero pixel'),
 'vpc_avenue_mat': (f'{S}/remote/vpc_avenue.mat', 'm3:/workspace/m3/vpc/data/avenue/avenue.mat (from VideoPatchCore Google-Drive avenue.zip, CRC da034f7e); MAT header: PCWIN64, created Fri Oct 23 20:15:43 2015; cell gt{21} of [start;end] 1-based inclusive intervals'),
 'accivad_avenue_mat': ('/home/lei/Desktop/msc/ext/Accurate-Interpretable-VAD/data/avenue.mat', 'AccI-VAD repo data/avenue.mat; MAT header: created Wed Jan 20 23:58:42 2021; per-frame int32 arrays keyed 01..21'),
 'aedmae_avenue_gt_txt': (f'{S}/remote/gt_files.zip', 'AED-MAE gt_files.zip/Avenue_gt/<vid>.txt (np.savetxt 0/1 per frame, zip dated 2022-03-17)'),
}
meta = {k: dict(source=v[1], **h(v[0])) for k, v in files.items()}
tot = {s: int(sum(v[k].sum() for k in ids)) for s, v in src.items()}
pair = {f'{a}__vs__{b}': int(sum(int((src[a][k] != src[b][k]).sum()) for k in ids)) for a, b in itertools.combinations(src, 2)}
res = dict(
  note='Frame indices are 1-based, inclusive. n_frames = decoded test frames (OpenCV), equal to len(volLabel) for all 21 videos.',
  generated='2026-11-05', total_test_frames=sum(L.values()), sources=meta, total_positive_frames=tot,
  pairwise_differing_frames=pair,
  identical_groups=[['ours_gt_json', 'cuhk_mask_anypixel', 'accivad_avenue_mat'], ['vpc_avenue_mat', 'aedmae_avenue_gt_txt']],
  cuhk_vs_vpc_summary=dict(
     n_differing=int(sum(per[k]['n_diff_cuhk_vs_vpc'] for k in ids)),
     only_cuhk_frames=int(sum(sum(e-s+1 for s, e in per[k]['only_cuhk']) for k in ids)),
     only_vpc_frames=int(sum(sum(e-s+1 for s, e in per[k]['only_vpc']) for k in ids)),
     videos_identical=[k for k in ids if per[k]['n_diff_cuhk_vs_vpc'] == 0],
     n_events_vpc=int(sum(per[k]['n_events_vpc_intervals'] for k in ids)),
     n_runs_cuhk=int(sum(per[k]['n_runs_cuhk'] for k in ids)),
     by_type={t: dict(segments=v[0], frames=v[1]) for t, v in sorted(cat_tot.items())},
     mask_pixels_in_only_cuhk_frames=dict(min=int(min(cuhk_only_area)), median=float(np.median(cuhk_only_area)), max=int(max(cuhk_only_area))),
     mask_pixels_all_cuhk_positive_frames=dict(min=int(min(all_pos_area)), median=float(np.median(all_pos_area)), max=int(max(all_pos_area)))),
  per_video=per)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ['total_positive_frames', 'pairwise_differing_frames', 'cuhk_vs_vpc_summary']}, indent=1))
for k in ids:
    p = per[k]
    if p['n_diff_cuhk_vs_vpc']:
        print(k, p['n_diff_cuhk_vs_vpc'], [(c['only_in'], c['frames'], c['type']) for c in p['diff_segments']])
