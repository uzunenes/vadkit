# vadkit design (v0.1, 4 Nov 2026)

Goal: one small package that reproduces our archived results bit-for-bit, then lets a YAML config swap every stage.
The rules are the paper's:
- strictly causal;
- normal training videos only for fitting and selection;
- one configuration for every dataset;
- no test-label selection.

## Layout and ownership

| Path | Contents | Must be equivalent to |
|---|---|---|
| `core/interfaces.py`, `core/registry.py` | ABCs, dataclasses, name registry (written, do not change signatures; one exception: `AttributeExtractor.raw_key` + `fit(rows, rng=None)` + `state()` / `from_state()`, the explicit hook of a normal-data fitted extractor) | – |
| `io/frames.py` | `frame_paths(root, split)` (numeric order + assert stems == range(n)), `read(path, width)`, `VideoSource` (dir / file / camera) | `tools/velpilot.py` `frame_paths`/`read` |
| `perception/detector.py` | `YoloMotionDetector`: YOLO boxes (conf 0.25) → `remove_covered` → motion-contour boxes from frames t-gap, t (stateful, causal) | `tools/objfeat2.py` boxes at t ≥ gap |
| `perception/pose.py` | `YoloPose.attach` (mutual cover > 0.5, keypoints relative to box centre / height) | `tools/objfeat2.py` pose / pose_ok |
| `perception/flow.py` | `DISFlow`, `RAFTFlow` (C_T_SKHT_V2), `NeuFlowFlow` (RGB, 768×432, rescaled) | `tools/velpilot.py` `Flow` |
| `perception/embed.py` | `OpenClipEncoder` (MobileCLIP2-S0 / dfndr2b, crop → S×S bilinear, normalise, L2, fp16) | `tools/objfeat2.py` app |
| `attributes/velocity.py` | `OrientationHist` (c8), `Magnitude` (c1), height-normalised variants | `tools/objfeat2.velocity`, `tools/velpilot.construct` |
| `attributes/pose.py`, `attributes/appearance.py` | pose rows (matched persons only), appearance PCA-64 fitted on normal rows (`raw_key` = its `key` column; state keeps `dim`, `max_rows`, `key`) | `tools/objscore2.py` |
| `density/niw.py` | `NIWDensity` + `fit_attribute(...)`: 2-fold split by sorted video name, held-out likelihood selection (`aean.niw.select_mixture`), cross-fitted reference, floor for frames without objects, frame score = max over objects of −log p; config `density: niw` or `{name: niw, args: {...}}` with args forwarded (fixed `K`, `kappa0`, `nu0` together, or the selection's `max_eval` / `min_rows`) | `tools/objscore2.py` / `tools/velpilot.py score` |
| `calibration/fisher.py` | `FisherCalibrator` | `tools/fuse_runs.normaliser(ref, "fisher")` |
| `calibration/gpd.py`, `calibration/hier.py` | HB-POT (stage 1 of `docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md`): `GPDTailCalibrator` (Fisher body, GPD tail; `fisher_gpd`), group calibrator `fisher_hb` (`fit_groups(refs, ids, mode)`, arms `t0` / `t1` / `t2`: GPD MLE, video-cluster bootstrap, REML empirical Bayes, truncated posterior mode); grid channels only | `t0` ≡ `FisherCalibrator` bit for bit |
| `fusion/sum.py` | `SumFusion` (all channels, or a fixed list from the config) | `tools/objfuse.py` sum |
| `temporal/median.py` | `CausalMedian(k=9)` streaming, plus `causal_median(arr, k)`; the only temporal stage: config `temporal` is `causal_median` with an integer `k` (default 9, the paper's protocol), anything else raises | `tools/objfuse.causal_median` |
| `eval/protocol.py` | `evaluate(scores, gt, warmup=4, smooth=causal_median)` → causal micro, causal macro, offline (the pipeline passes `causal_median(·, k)` of the config, the k its stream uses); paired video bootstrap | `tools/objfuse.evaluate`, `tools/bootstrap_fused.py` |
| `pipeline/offline.py` | object branch: extract → feat2-format npz cache → fit → score → calibrate → fuse → evaluate; also loads released / pilot npz. Per-frame `Perception` runs the flow at batch 1 (unless the spec sets `batch`); `extract` batches the flow pairs of the frames with objects in the back-end's batch (velpilot flowvel), defaults `train_every` to the config's, takes `shard` `i/n`. `FittedState` records `gap` (the uniform startup of its training features) and `channels` (attributes fitted in every group); `score` writes those channels, one file per scene with `--scenes` | archived `obj2` results |
| `pipeline/stream.py` | `StreamingVAD.step(bgr) -> FrameResult` using the fitted state: its `channels` (attributes fitted in every group, as scored and fused offline) at its `gap` (startup of the training features) | offline scores on the same video |
| `grid/precomputed.py` | adapter for archived cell channels (`I`, `Fs`, `Fd` scores.npz + ref) so fusion can include the grid branch now (a `{sc}` template with no file for any scene raises, one missing some scenes warns; order without scenes: `I`, `Fs`, `Fd`, other grid keys, object channels); calibration in two passes (every group's channels loaded, then each channel calibrated per group or by a group calibrator over all groups); the grid calibrator is the config's `grid_calibrator` (default: `calibrator`) or `--grid-calibrator`, a group calibrator needs `--extra-ref-ids` (`tools/grid_ref_ids.py`) and raises without them; a native streaming grid branch wrapping `aean.scoring` comes in round 2 | `tools/objfuse.py` channels |
| `plugins.py`, `cli.py`, `configs/*.yaml` | registration of built-ins, `python -m vadkit {extract,fit,score,eval,bench}` | – |

## Golden tests (`tests/vadkit/`, skipped when data are absent)

Tests over 60 s on a CPU (model ports, YOLO / open_clip agreement, the full held-out selection on Ped2) carry the
`slow` marker (`tests/vadkit/conftest.py`) and run only with `VADKIT_SLOW=1`.

Ped2 data on the laptop:
- `/home/lei/Desktop/msc/data/frames/ped2`
- `/home/lei/Desktop/msc/data/feat2tmp/feat2/ped2` (released light features)
- `/home/lei/Desktop/msc/data/cache/ped2/gt.json`

The tests:

1. Calibration, fusion, causal median and evaluate equal the reference functions on random inputs (exact).
2. `fit_attribute` on the released Ped2 features reproduces the archived causal AUCs: ovel 0.8616, opose 0.5619, oapp 0.9424 (`sonuclar/obj2/ped2/fuse.txt`). The rng sequence of `objscore2.py` is shared across attributes in the order ovel, opose, oapp.
3. DIS + `OrientationHist` on the released Ped2 boxes reproduces the released `vel` arrays bit for bit.
4. `YoloMotionDetector` on Ped2 reproduces the released box counts on ≥ 98% of frames. The released boxes came from a GPU, so CPU numerics may differ.
5. Streaming equals offline: per-frame fused scores of one Ped2 test video agree to 1e-9.
