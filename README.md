# vadkit: light, modular and online video anomaly detection for fixed cameras

This repository holds the code and the evidence of the paper

> E. Uzun and K. Kayabol, "Light, Modular and Online: Object Attributes and Bayesian Cell Statistics for Video
> Anomaly Detection," under review at *IEEE Access*.

It contains:

- the `vadkit` package (the object branch, the fusion and the evaluation protocols);
- the research code of the grid branch;
- the configuration file of every experiment;
- the scripts that produce the tables and figures of the paper;
- the evidence files that every reported number is generated from (results, timing and size measurements, logs);
- the pre-registration documents.

The detector is meant for fixed cameras. It learns from normal video of the scene only, scores every frame when it
arrives (causal, with a delay of up to four frames from the median filter) and needs no test labels for any choice.

## Method in one paragraph

**Object branch.**
- Small off-the-shelf networks describe every object in a frame: YOLO11n boxes plus motion-contour boxes,
  YOLO11n-pose keypoints, DIS optical flow and MobileCLIP2-S0 crop embeddings.
- Each object gets three attributes: a velocity histogram, its pose and a 64-dimensional appearance code.
- Each attribute is scored with a normal-inverse-Wishart (NIW) mixture density. The hyper-parameters of the density
  are selected on held-out normal video.
- This branch trains no network.

**Grid branch (optional).**
- A small autoencoder is trained per scene. Its residual and the optical flow are scored in fixed image cells.
- The cell statistics are shrunk towards the scene, following the hyperspectral background models of Arısoy and
  Kayabol.

**Fusion.** Every channel is calibrated against its own normal reference. The calibrated channels are summed and
smoothed with a causal median of 9 frames.

## Results

Frame-level micro AUC (%):
- "standard" = per-video min-max normalisation and Gaussian smoothing, as most published results use;
- "online" = causal median, no statistics of the test video.

Ped1 was the development set.

| Configuration | Parameters | Ped1 std / online | Ped2 std / online | Avenue std / online | ShanghaiTech std / online |
|---|---|---|---|---|---|
| Object branch | 16.9 M | 74.6 / 70.9 | 89.5 / 93.9 | 87.9 / 82.8 | 78.4 / 79.5 |
| Object + grid | 17.9 M | 79.4 / 76.4 | 96.7 / 96.5 | 89.8 / 87.4 | 76.9 / 70.2 |

- **Speed:** with TensorRT detectors the object branch runs end to end at 25.7 frames/s on an RTX 3090, at batch
  size 1.
- **Scope:** we make no claim about embedded hardware.
- **Source:** every value comes from the evidence files with the scripts below (`gen/numbers_v3.tex`,
  `gen/numbers_text_v3.tex`).

## Installation

Python 3.11 or newer (the paper's runs used 3.11; the tests also pass on 3.14).

```bash
git clone https://github.com/uzunenes/vadkit.git
cd vadkit
python -m venv .venv && . .venv/bin/activate
pip install torch torchvision          # choose the CUDA or CPU build for your machine, see pytorch.org
pip install -r requirements.txt
```


`requirements.txt` lists the packages with the versions of the measured runs.

The paper's runs used Python 3.11.9, torch 2.4.1+cu121, torchvision 0.19.1 and the versions above (full lists:
`paper/ieee_access/kanit/**/pip_freeze.txt`, `env.json`). The table and figure scripts need only the standard library
and, for the figures, `matplotlib`. The tests need `pytest`.

**Weights downloaded on the first run:**
- YOLO11n and YOLO11n-pose, from the Ultralytics release assets;
- MobileCLIP2-S0, from the Hugging Face hub.

These models and the `ultralytics` package keep their own licenses. Ultralytics is AGPL-3.0, so check them before you
use this code in a product.

## Run the object branch on your own camera

**1. Put normal training video as frames.** The layout is `<root>/training/frames/<video>/<t>.jpg`, with
`t = 0 … n-1` (time order).

**2. Make a config with your frame gap.** Copy `vadkit/configs/r2/base.yaml` and set `gap:` to an integer. The gap is
the number of frames between the two images of the optical flow: we used 1 at 10 fps (UCSD) and 2 at 25 fps
(Avenue, ShanghaiTech).

**3. Extract the features and fit the densities.** This trains no network.

```bash
python -m vadkit extract --config mycam.yaml --frames <root> --splits training --out feat/mycam
python -m vadkit fit     --config mycam.yaml --features feat/mycam --out state/mycam
```

**4. Score a video, a frame folder or a camera,** one frame at a time:

```python
from vadkit.io.frames import VideoSource
from vadkit.pipeline import FittedState, StreamingVAD

vad = StreamingVAD(FittedState.load("state/mycam"))
for t, frame in VideoSource("test.mp4"):     # also a frame folder or a camera index such as "0"
    r = vad.step(frame)                      # r.score is None during the first 4 frames
    if r.score is not None:
        print(t, r.score, r.channels)        # fused online score and the calibrated channels
```

**Notes:**
- `--device cpu` forces the CPU.
- The alarm threshold is not calibrated without labels (Section "Limitations" of the paper). You have to set it per
  camera.


## Reproduce the paper

### Tables and figures from the evidence files

Run these from the repository root. No GPU and no dataset are needed.

```bash
python3 paper/ieee_access/fig_src/make_tables_v3.py      # main, online, size, speed and setup tables, numbers_v3.tex
python3 paper/ieee_access/fig_src/make_hakem_v3.py       # scoring ablation, hyper-parameters, latency tables
python  paper/ieee_access/fig_src/make_text_v3.py        # numbers of the running text, ablation table
python  paper/ieee_access/fig_src/fig_size_accuracy.py   # fig/size_accuracy.png   (needs matplotlib)
python  paper/ieee_access/fig_src/fig_data_efficiency.py # fig/data_efficiency.png (needs matplotlib)
git diff --exit-code paper/ieee_access/gen                # the committed tables are reproduced byte for byte
```

The PNG bytes of the figures can differ with another matplotlib version.

**Rules of the scripts:**
- Each script reads only `paper/ieee_access/kanit/` (the evidence), `fig_src/tables_v3_inputs.json` and a few code,
  config and log files.
- It stops on a missing or inconsistent input; there is no silent fallback.
- It writes a provenance file next to the tables (`gen/*_provenance.txt`). This file gives the source file and field
  of every number, and for the reviewer-driven measurements also the sha256 of every input.

**`make_text_v3.py`** writes the numbers of the running text (`gen/numbers_text_v3.tex`, `gen/tab_ablation_v3.tex`,
`gen/tab_online_v4.tex`); run it with `python paper/ieee_access/fig_src/make_text_v3.py`. In the development
repository it also checks the pre-registration commits against the manuscript and the dated git history; neither is
in this repository, so here it skips that check and says so.

### From the datasets

GPU recommended.

**1. Frame labels and frame folders.**

```bash
python prepare_ucsd.py   --root UCSD_Anomaly_Dataset.v1p2 --out cache --peds UCSDped2 UCSDped1
python prepare_avenue.py --zip Avenue_Dataset.zip --gt-zip ground_truth_demo.zip --out cache/avenue --splits test
python vad_bench/vast/runs_velpilot/gt_sht.py --root shanghaitech --out cache/shanghaitech/gt.json
python vad_bench/vast/hazirla_frames.py --ds ped2 --src UCSD_Anomaly_Dataset.v1p2 --out frames --gt cache/ped2/gt.json
```

**2. Object branch.** The configs are in `vadkit/configs/r2/`. For ShanghaiTech, use the `*_sht.yaml` config, add
`--scenes` to `eval`, and add `--train-every 5` to `extract`.

```bash
python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames frames/ped2 --dataset ped2 --out feat/base/ped2
python -m vadkit eval    --config vadkit/configs/r2/base.yaml --features feat/base/ped2 --gt cache/ped2/gt.json \
    --out res/base/ped2_obj.json
```

**3. Grid branch.**
- Train an autoencoder per scene: `train.py --variant 2d --no-adv`.
- Score the cell channels: `evaluate.py --pool 48 --no-state`, with the settings of `mc4.sh`, `mc5.sh` and
  `shtech.sh`.
- Pass the resulting `scores.npz` files to `vadkit eval --extra-channels`:

```bash
python -m vadkit eval --config vadkit/configs/r2/base.yaml --features feat/base/ped2 --gt cache/ped2/gt.json \
    --extra-channels runs/ped2_2d_noadv/eval_ped2_pool48_mc4/scores.npz:frx_bayes=I \
                     runs/ped2_2d_noadv/eval_ped2_pool48_mc5/scores.npz:flow_ubm_m=Fs,flow_rx_m=Fd \
    --out res/base/ped2_grid.json
```

**The exact commands of every experiment** are in the run scripts:
- component study: `vad_bench/vast/runs_r2/run.sh`;
- scoring ablation, latency and setup measurements, pose rule: `vad_bench/vast/runs_hakem/`;
- hierarchical calibration: `vad_bench/vast/runs_hb/`;
- velocity pilot: `vad_bench/vast/runs_velpilot/`.

The scripts were written for a rented GPU machine with the repository at `/workspace/cctv_aean`, so adapt the paths.

### Tests

```bash
python -m pytest -q vad_bench/tests tests
```

**Tests that need data:**
- The golden tests compare against our archived Ped2 results. They need the Ped2 frames, the Ped2 labels and the Ped2
  features of our runs, and the features are not distributed. These tests are skipped when the data are missing.
- Point to the data with `VADKIT_PED2_FRAMES`, `VADKIT_PED2_FEAT`, `VADKIT_PED2_FEAT2` and `VADKIT_PED2_GT`.
- Tests that take more than a minute on a CPU run only with `VADKIT_SLOW=1`.

## Repository layout

| Path | Contents |
|---|---|
| `vadkit/` | The package: perception, attributes, NIW density, calibration, fusion, causal median, evaluation protocols, offline and streaming pipelines, CLI (`python -m vadkit {extract,fit,score,eval,bench}`). Design notes: `vadkit/DESIGN.md` |
| `vadkit/configs/` | `r2/*.yaml`: the configurations of the paper (`base` is the reported one; the others are the component study and the TensorRT and no-pose variants) |
| `aean/`, `train.py`, `evaluate.py`, `prepare_*.py`, `mc*.sh`, `shtech.sh` | Grid branch: autoencoders, cell statistics (FRX, FRX-Bayes, optical-flow GMM-UBM and RX), dataset caches, the runs that produced the grid channels |
| `tools/` | Analysis scripts: fusion and channel selection, bootstrap, threshold evaluation, timing, velocity pilot |
| `vad_bench/` | Common evaluator, protocol notes (`MATCHED_PROTOCOL.md`, `SPEED_PROTOCOL.md`), run scripts of every experiment (`vast/runs_*`), tests |
| `tests/vadkit/` | Golden tests of the package against the archived results |
| `paper/ieee_access/fig_src/` | Scripts that write the tables, number macros and figures; `tables_v3_inputs.json` holds the published competitor values with verbatim quotes and locations |
| `paper/ieee_access/gen/` | Generated tables, macros and provenance files |
| `paper/ieee_access/kanit/` | Evidence files ("kanıt"): every number of the paper is generated from here |
| `docs/tez_notlari/` | Pre-registration documents (`*_on_kayit_*.md`) and their result notes |
| `scores/` | Per-frame scores of our detector on the test videos |

**Evidence folders** (the names are Turkish):
- `r2_20261105` is the component study, the grid-branch bootstrap and the TensorRT test sets.
- `hakem_20261106` holds the measurements added after review: the scoring ablation, the latency and setup times and
  the pose rule.
- `hiz_20261105` is speed.
- `kurulum_20261105` is setup cost, data efficiency and thresholds.
- `boyut_20261105` is model size.
- `hb_20261105` is the hierarchical tail calibration.
- `avenue_etiket_20261105` compares the Avenue label sets.
- `literature` holds the published competitor values with the sources we checked them against.

## Pre-registration

The experiments that the paper calls pre-registered were specified in these documents before their test runs: the
hypotheses, the success criteria and the selection rules. The documents are written in Turkish.

| Document (`docs/tez_notlari/`) | Commit | Date (UTC+3) | Content |
|---|---|---|---|
| `tur2_on_kayit_2026-11-05.md` | `3dffc5a` | 5 Nov 2026, 08:52 | Component study with latency-budget rule, grid-branch ablation |
| `kurulum_olcumleri_on_kayit_2026-11-05.md` | `e4031e4` | 5 Nov 2026, 11:41 | Data efficiency, alarm thresholds, feature-extraction time estimate |
| `hiz_on_kayit_2026-11-05.md` | `2921bc5` | 5 Nov 2026, 15:41 | TensorRT equivalence check |
| `hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md` | `6b08c14` | 5 Nov 2026, 16:17 | Hierarchical tail calibration of the grid channels |
| `hakem_duzeltmeleri_on_kayit_2026-11-06.md` | `4828608` (+ `3d854f1`) | 6 Nov 2026, 10:10 (10:11) | Scoring ablation, grid-branch latency, latency spread, setup times |
| `kiss1_hiz_buyuklugu_on_kayit_2026-11-06.md` | `2b7ae32` | 6 Nov 2026, 12:24 | Velocity-magnitude attribute |
| `e1_poz_kapsama_kapisi_on_kayit_2026-11-06.md` | `1ace435` | 6 Nov 2026, 12:48 | Pose coverage rule |
| `e1_pozsuz_hiz_olcumu_on_kayit_2026-11-06.md` | `971391c` | 6 Nov 2026, 15:46 | Speed and fitting time without the pose model |

The commit hashes refer to the private development repository. It also holds third-party material, so it is not
public. **Its dated git history is available to the editors and reviewers on request.**

## Datasets (not included)

This repository does not redistribute the following:
- dataset videos or frames;
- extracted features;
- trained autoencoders;
- third-party model weights.

Download the datasets from their authors and keep to their terms:

- **UCSD Ped1 and Ped2:** http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- **CUHK Avenue:** http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html (with `ground_truth_demo.zip`
  for the frame labels)
- **ShanghaiTech Campus:** https://svip-lab.github.io/dataset/campus_dataset.html


## Citation

The paper is under review at *IEEE Access*. Until it is published, please cite:

```bibtex
@misc{uzun2026vadkit,
  author = {Uzun, Enes and Kayabol, Koray},
  title  = {Light, Modular and Online: Object Attributes and Bayesian Cell Statistics for Video Anomaly Detection},
  note   = {Under review at IEEE Access},
  year   = {2026},
  url    = {https://github.com/uzunenes/vadkit}
}
```

## License

MIT, see `LICENSE`. The datasets, the pretrained models that the code downloads, and the libraries keep their own
licenses.

## Acknowledgements

This is MSc thesis work at the Department of Electronics Engineering, Gebze Technical University, supervised by
Prof. Koray Kayabol. Ford Otosan funded the article processing charge.
