"""Stage 3 experiments:
(a) ShanghaiTech train/test scene overlap audit (scenes 01..12 in both splits; scene 13 train-only).
(b1) f = 0.10 data-efficiency comparison: NIW mixture (D1) vs. ML-GMM (D2), prior-free same-components (D2b), and 1-NN (D4)
     across all four datasets (5 draws, seeds 0..4, matching kanit/kurulum_20261105/dataeff_<ds>_0.1.json).
(b2) Leave-one-scene-out cross-scene prior transfer on ShanghaiTech across the 12 overlapping scenes (01..12) at
     f = 0.10, f = 0.25, and f = 1.00.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from aean.niw import NIWCells, NIWMixture, _kmeans
from vadkit.calibration.fisher import FisherCalibrator
from vadkit.density.niw import AttributeFit, NIWDensity, VideoRows, _rows, scene_of
from vadkit.eval.protocol import evaluate, paired_bootstrap
from vadkit.pipeline.offline import ObjectBranch, channel_order, load_config, record_rows

ROOT = Path(__file__).resolve().parents[1]
FEAT_ROOT = Path(os.environ.get("FEAT_ROOT", "data/r2_feat/base"))
GT_ROOT = Path(os.environ.get("GT_ROOT", "data/cache"))
DATAEFF = ROOT / "paper/ieee_access/kanit/kurulum_20261105"
S1_DIR = ROOT / "paper/ieee_access/kanit/hakem_20261106/s1"

_spec = importlib.util.spec_from_file_location("ablation", ROOT / "vad_bench/vast/runs_hakem/ablation.py")
_abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_abl)
subset = _abl.subset
alt_calibrated = _abl.alt_calibrated
niw_calibrated = _abl.niw_calibrated
fused = _abl.fused
vrows = _abl.vrows


def sht_scene_overlap() -> dict:
    sht_dir = FEAT_ROOT / "shanghaitech"
    tr_files = sorted((sht_dir / "training").glob("*.npz"))
    te_files = sorted((sht_dir / "testing").glob("*.npz"))
    tr_c = Counter(scene_of(f.stem) for f in tr_files)
    te_c = Counter(scene_of(f.stem) for f in te_files)
    all_scenes = sorted(set(tr_c) | set(te_c))
    overlap = [sc for sc in all_scenes if tr_c[sc] > 0 and te_c[sc] > 0]
    train_only = [sc for sc in all_scenes if tr_c[sc] > 0 and te_c[sc] == 0]
    test_only = [sc for sc in all_scenes if tr_c[sc] == 0 and te_c[sc] > 0]
    per_scene = {
        sc: {"train_videos": int(tr_c[sc]), "test_videos": int(te_c[sc]), "in_both": sc in overlap}
        for sc in all_scenes
    }
    return {
        "total_scenes": len(all_scenes),
        "train_scenes": len(tr_c),
        "test_scenes": len(te_c),
        "overlap_scenes": overlap,
        "n_overlap_scenes": len(overlap),
        "train_only_scenes": train_only,
        "test_only_scenes": test_only,
        "total_train_videos": sum(tr_c.values()),
        "overlap_train_videos": sum(tr_c[sc] for sc in overlap),
        "total_test_videos": sum(te_c.values()),
        "per_scene": per_scene,
    }


def run_f010_all_datasets() -> dict:
    results = {}
    for ds in ("ped1", "ped2", "avenue", "shanghaitech"):
        t0 = time.perf_counter()
        scenes = ds == "shanghaitech"
        cfg = load_config(ROOT / ("vadkit/configs/r2/base_sht.yaml" if scenes else "vadkit/configs/r2/base.yaml"))
        gt = json.loads((GT_ROOT / ds / "gt.json").read_text())
        vids = sorted(gt)
        branch = ObjectBranch(cfg, scenes, "cpu")
        train, test = branch.load(FEAT_ROOT / ds, ("training", "testing"))
        de01 = json.loads((DATAEFF / f"dataeff_{ds}_0.1.json").read_text())
        s1_full = json.loads((S1_DIR / f"{ds}.json").read_text())
        draws = []
        for seed in range(5):
            keep = subset(train, scenes, 0.1, seed)
            st = branch.fit({v: train[v] for v in keep})
            tested = [v for v in vids if not st.scenes or scene_of(v) in st.groups]
            g = {v: gt[v] for v in tested}
            ch = list(channel_order({}, {c: None for c in st.channels}, scenes))
            zz = fused(niw_calibrated(branch, st, test, tested), ch)
            arch = [r for r in de01["rows"] if r["seed"] == seed][0]
            r_d1 = evaluate({v: zz[v] for v in tested}, g)
            row = {
                "seed": seed,
                "videos": len(keep),
                "minutes": arch["minutes"],
                "test_videos": len(tested),
                "channels_fitted": ch,
                "D1": r_d1,
                "D1_archived_causal": arch["causal"],
                "D1_abs_diff": abs(r_d1["causal"] - arch["causal"]),
            }
            for code in ("D2", "D2b", "D4"):
                zc = fused(alt_calibrated(code, st, train, keep, test, tested), ch)
                row[code] = evaluate({v: zc[v] for v in tested}, g)
            draws.append(row)
            print(
                f"[f=0.10] {ds} seed={seed}: D1={row['D1']['causal']*100:.2f}% "
                f"(arch diff={row['D1_abs_diff']:.1e}) D2={row['D2']['causal']*100:.2f}% "
                f"D2b={row['D2b']['causal']*100:.2f}% D4={row['D4']['causal']*100:.2f}%",
                flush=True,
            )

        summary = {"minutes_mean": float(np.mean([r["minutes"] for r in draws]))}
        for code in ("D1", "D2", "D2b", "D4"):
            c_vals = [r[code]["causal"] for r in draws]
            m_vals = [r[code]["causal_macro"] for r in draws]
            o_vals = [r[code]["offline"] for r in draws]
            full_c = s1_full["rows"][code]["fused"]["causal"]
            summary[code] = {
                "causal_mean": float(np.mean(c_vals)),
                "causal_std": float(np.std(c_vals, ddof=1)),
                "macro_mean": float(np.mean(m_vals)),
                "offline_mean": float(np.mean(o_vals)),
                "full_causal": float(full_c),
                "diff_vs_own_full": float(np.mean(c_vals) - full_c),
            }
        summary["D1_minus_D2_causal_mean"] = float(
            summary["D1"]["causal_mean"] - summary["D2"]["causal_mean"]
        )
        summary["D1_minus_D2b_causal_mean"] = float(
            summary["D1"]["causal_mean"] - summary["D2b"]["causal_mean"]
        )
        summary["elapsed_s"] = time.perf_counter() - t0
        results[ds] = {"draws": draws, "summary": summary}
    return results


def finalize_with_external_prior(
    cells: NIWCells,
    kappa0: float,
    nu0: float,
    mu0_ext: torch.Tensor,
    Sigma0_ext: torch.Tensor,
    beta: float = 1.0,
    ridge: float = 1e-3,
) -> NIWCells:
    """Finalize NIWCells where the prior center (mu0, Sigma0) blends scene statistics with cross-scene statistics
    from the other 11 scenes (beta=1.0: pure cross-scene prior center; beta in (0,1): convex shrinkage)."""
    C = cells.C
    N = cells.n.sum()
    mu_self = cells.s1.sum(0) / N
    Sigma_self = cells.s2.sum(0) / N - torch.outer(mu_self, mu_self)
    mu0 = (1.0 - beta) * mu_self + beta * mu0_ext
    d0 = mu_self - mu0_ext
    Sigma0 = (1.0 - beta) * Sigma_self + beta * Sigma0_ext + beta * (1.0 - beta) * torch.outer(d0, d0)
    Sigma0 = Sigma0 + (ridge * Sigma0.diagonal().mean() + 1e-12) * torch.eye(
        C, dtype=Sigma0.dtype, device=Sigma0.device
    )
    nu0 = max(float(nu0), C + 1.0 + 1e-3)
    Psi0 = (nu0 - C - 1) * Sigma0
    n = cells.n[:, None]
    xbar = cells.s1 / n.clamp(min=1e-12)
    S = cells.s2 - n[..., None] * xbar[:, :, None] * xbar[:, None, :]
    kn = kappa0 + cells.n
    nun = nu0 + cells.n
    mun = (kappa0 * mu0 + cells.s1) / kn[:, None]
    d = xbar - mu0
    Psin = Psi0 + S + (kappa0 * cells.n / kn)[:, None, None] * d[:, :, None] * d[:, None, :]
    dof = nun - C + 1
    scale = Psin * ((kn + 1) / (kn * dof))[:, None, None]
    L = torch.linalg.cholesky(scale)
    cells.mu, cells.L, cells.dof = mun, L, dof
    cells.logdet = 2 * torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(-1)
    cells.lognorm = (
        torch.lgamma((dof + C) / 2)
        - torch.lgamma(dof / 2)
        - 0.5 * C * np.log(np.pi)
        - 0.5 * C * torch.log(dof)
        - 0.5 * cells.logdet
    )
    cells.kappa0, cells.nu0 = kappa0, nu0
    return cells


def fit_niw_mixture_ext(
    X: torch.Tensor,
    K: int,
    kappa0: float,
    nu0: float,
    mu0_ext: torch.Tensor,
    Sigma0_ext: torch.Tensor,
    beta: float = 0.5,
) -> NIWMixture:
    X = X.to("cpu", torch.float64)
    K_eff = min(int(K), max(1, len(X) // 2))
    m = NIWMixture(K_eff, kappa0, nu0, device="cpu")
    m.centres = _kmeans(X, K_eff, seed=0) if K_eff > 1 else X.mean(0, keepdim=True)
    a = m.assign(X)
    cells = NIWCells(X.shape[1], K_eff, "cpu").update(X, a)
    m.cells = finalize_with_external_prior(cells, kappa0, nu0, mu0_ext, Sigma0_ext, beta=beta)
    n = m.cells.n
    m.logw = torch.log((n + m.alpha0 / K_eff) / (n.sum() + m.alpha0))
    return m


def run_cross_scene_prior_sht(overlap_scenes: list[str]) -> dict:
    """Leave-one-scene-out cross-scene transfer on the 12 overlapping ShanghaiTech scenes (01..12):
    For each target scene g in 01..12, the 'donor' scenes are the other 11 overlapping scenes (01..12 \\ {g}).
    We compare:
      1. scene_cv (default): per-scene 2-fold CV hyperparameter selection + scene-own prior center.
      2. loo_hyper: donor-scenes modal/median hyperparameters (K, kappa0, nu0) transferred to target scene g
         (skipping per-scene 2-fold CV grid search) + scene-own prior center.
      3. loo_prior_and_hyper: donor-scenes global prior center (mu0, Sigma0) + donor hyperparameters transferred
         to target scene g (for velocity and pose, which share a fixed coordinate system across scenes; appearance
         uses donor PCA + donor prior center).
    Evaluated at full data (f = 1.0), quarter data (f = 0.25, 5 draws), and 10% data (f = 0.10, 5 draws).
    """
    cfg = load_config(ROOT / "vadkit/configs/r2/base_sht.yaml")
    gt = json.loads((GT_ROOT / "shanghaitech/gt.json").read_text())
    vids = sorted(gt)
    branch = ObjectBranch(cfg, True, "cpu")
    train_all, test = branch.load(FEAT_ROOT / "shanghaitech", ("training", "testing"))
    # Restrict strictly to the 12 overlapping scenes (01..12)
    train = {v: r for v, r in train_all.items() if scene_of(v) in overlap_scenes}
    every = int(cfg.get("train_every", 5))
    drop = bool(cfg.get("drop_startup", False))

    # First fit full state on the 12 overlapping scenes to get each scene's selected hyperparameters and fit time
    t0 = time.perf_counter()
    full_state = branch.fit(train)
    full_cv_fit_s = time.perf_counter() - t0

    # Donor modal hyperparameters across the other 11 scenes for each scene g and attribute a
    def donor_hyper(target_sc: str, attr: str) -> dict:
        ks, k0s, nu0s = [], [], []
        for sc in overlap_scenes:
            if sc == target_sc:
                continue
            if attr in full_state.groups[sc]:
                h = full_state.groups[sc][attr].hyper
                ks.append(int(h["K"]))
                k0s.append(float(h["kappa0"]))
                nu0s.append(float(h["nu0"]))
        # Median in log/sorted space
        ks.sort()
        k0s.sort()
        nu0s.sort()
        m = len(ks) // 2
        return {"K": ks[m], "kappa0": k0s[m], "nu0": nu0s[m]}

    # Evaluate loo_hyper on full training set (f = 1.0) and measure fitting time without CV
    t0 = time.perf_counter()
    z_loo_full = {a: {} for a in ("ovel", "opose", "oapp")}
    for sc in overlap_scenes:
        tr_sc = {v: r for v, r in train.items() if scene_of(v) == sc}
        te_sc = {v: test[v] for v in vids if scene_of(v) == sc}
        for a in ("ovel", "opose", "oapp"):
            ext = full_state.groups[sc][a].extractor
            vr_tr = {v: vrows(ext, r) for v, r in tr_sc.items()}
            vr_te = {v: vrows(ext, r) for v, r in te_sc.items()}
            dh = donor_hyper(sc, a)
            af = _abl.fit_attribute if hasattr(_abl, "fit_attribute") else None
            from vadkit.density.niw import fit_attribute
            fit_res = fit_attribute(vr_tr, vr_te, np.random.default_rng(0), "cpu", every, drop, dh)
            if fit_res is not None:
                cal = FisherCalibrator().fit(fit_res.ref)
                for v, s in fit_res.scores.items():
                    z_loo_full[a][v] = cal(s)
    loo_hyper_fit_s = time.perf_counter() - t0

    chans = ["ovel", "opose", "oapp"]
    z_base_full = niw_calibrated(branch, full_state, test, vids)
    f_base_full = fused(z_base_full, chans)
    f_loo_full = fused(z_loo_full, chans)
    eval_base_full = evaluate(f_base_full, gt)
    eval_loo_full = evaluate(f_loo_full, gt)
    boot_loo_full = paired_bootstrap(f_loo_full, f_base_full, gt, 4, 2000, 0)["difference"]
    print(
        f"[SHT LOO f=1.0] base_cv={eval_base_full['causal']*100:.2f}% ({full_cv_fit_s:.2f}s) "
        f"loo_hyper={eval_loo_full['causal']*100:.2f}% ({loo_hyper_fit_s:.2f}s, diff={boot_loo_full['point']*100:+.2f}pt)",
        flush=True,
    )

    # Precompute donor prior centers for each (scene, attribute) pair in advance
    t_pc0 = time.perf_counter()
    donor_centers = {}
    for sc in overlap_scenes:
        tr_donors = {v: r for v, r in train.items() if scene_of(v) != sc}
        for a in chans:
            ext = full_state.groups[sc][a].extractor
            X_don_list = [_rows(vrows(ext, r), True, every, drop)[1] for r in tr_donors.values()]
            X_don = torch.from_numpy(np.concatenate([x for x in X_don_list if len(x)], axis=0).astype(np.float64))
            mu0_don = X_don.mean(0)
            Sigma0_don = (X_don.T @ X_don) / len(X_don) - torch.outer(mu0_don, mu0_don)
            donor_centers[(sc, a)] = (mu0_don, Sigma0_don)
    print(f"Precomputed donor centers for 12 scenes in {time.perf_counter() - t_pc0:.2f}s", flush=True)

    # Now test at f = 0.10 and f = 0.25 (5 draws each):
    # Compare:
    #   - scene_cv (default NIW with per-scene CV on the small subset)
    #   - loo_hyper (transferred (K, kappa0, nu0) from the other 11 scenes, no CV needed)
    #   - loo_prior_blend (transferred (K, kappa0, nu0) + donor prior center blended with beta=0.5)
    frac_results = {}
    for frac in (0.10, 0.25):
        draws_out = []
        for seed in range(5):
            keep = [v for v in subset(train_all, True, frac, seed) if scene_of(v) in overlap_scenes]
            tr_sub = {v: train[v] for v in keep}
            # 1) Default per-scene CV
            t_cv0 = time.perf_counter()
            st_cv = branch.fit(tr_sub)
            t_cv = time.perf_counter() - t_cv0
            z_cv = niw_calibrated(branch, st_cv, test, vids)
            f_cv = fused(z_cv, list(st_cv.channels))
            ev_cv = evaluate(f_cv, gt)

            # 2) LOO hyperparameter transfer + 3) LOO prior-center + hyperparameter transfer
            t_tr0 = time.perf_counter()
            z_lh = {a: {} for a in chans}
            z_lp = {a: {} for a in chans}
            for sc in overlap_scenes:
                tr_sc = {v: r for v, r in tr_sub.items() if scene_of(v) == sc}
                te_sc = {v: test[v] for v in vids if scene_of(v) == sc}
                for a in chans:
                    ext = full_state.groups[sc][a].extractor
                    vr_tr = {v: vrows(ext, r) for v, r in tr_sc.items()}
                    vr_te = {v: vrows(ext, r) for v, r in te_sc.items()}
                    dh = donor_hyper(sc, a)
                    names = sorted(vr_tr)
                    X_sc = {v: _rows(vr_tr[v], True, every, drop)[1] for v in names}
                    folds = (names[0::2], names[1::2])
                    fold_lens = [sum(len(X_sc[v]) for v in f) for f in folds]
                    if min(fold_lens) < 2:
                        # Fall back to st_cv if fewer than 2 rows in a fold
                        if a in z_cv and any(v in z_cv[a] for v in te_sc):
                            for v in te_sc:
                                z_lh[a][v] = z_cv[a][v]
                                z_lp[a][v] = z_cv[a][v]
                        continue
                    mu0_don, Sigma0_don = donor_centers[(sc, a)]

                    # Variant lh: donor hyper, scene own prior center (min_rows=2)
                    from vadkit.density.niw import fit_attribute
                    K_lh = max(1, min(dh["K"], min(fold_lens) // 2))
                    af_lh = fit_attribute(
                        vr_tr, vr_te, np.random.default_rng(0), "cpu", every, drop,
                        {"K": K_lh, "kappa0": dh["kappa0"], "nu0": dh["nu0"], "min_rows": 2}
                    )
                    if af_lh is not None:
                        cal_lh = FisherCalibrator().fit(af_lh.ref)
                        for v, s in af_lh.scores.items():
                            z_lh[a][v] = cal_lh(s)

                    # Variant lp: donor hyper + donor prior center (beta=0.5)
                    K_use = max(1, min(dh["K"], min(fold_lens) // 2))
                    ref_lp, keep_lp = [], []
                    for fit_n, ev_n in (folds, folds[::-1]):
                        Xf = torch.from_numpy(np.concatenate([X_sc[v] for v in fit_n], axis=0).astype(np.float64))
                        m_lp = fit_niw_mixture_ext(Xf, K_use, dh["kappa0"], dh["nu0"], mu0_don, Sigma0_don, beta=0.5)
                        for v in ev_n:
                            r = vr_tr[v]
                            f_idx, X_ev = _rows(r, True, every, drop)
                            s = np.full(int(r.n_frames), -np.inf)
                            if len(X_ev):
                                np.maximum.at(s, f_idx, -m_lp.logpdf(torch.from_numpy(np.asarray(X_ev, np.float64))).numpy())
                            ref_lp.append(s)
                            t = np.arange(int(r.n_frames))
                            keep_lp.append((t % every == 0) & (t >= int(r.startup) if drop else True))
                    ref_lp_arr = np.concatenate(ref_lp)[np.concatenate(keep_lp)]
                    floor_lp = float(ref_lp_arr[np.isfinite(ref_lp_arr)].min())
                    ref_lp_arr = np.where(np.isfinite(ref_lp_arr), ref_lp_arr, floor_lp)
                    cal_lp = FisherCalibrator().fit(ref_lp_arr)
                    X_all_sc = torch.from_numpy(np.concatenate([X_sc[v] for v in names], axis=0).astype(np.float64))
                    full_lp = fit_niw_mixture_ext(X_all_sc, K_use, dh["kappa0"], dh["nu0"], mu0_don, Sigma0_don, beta=0.5)
                    for v, r in vr_te.items():
                        f_idx, X_ev = _rows(r, False, every, drop)
                        s = np.full(int(r.n_frames), -np.inf)
                        if len(X_ev):
                            np.maximum.at(s, f_idx, -full_lp.logpdf(torch.from_numpy(np.asarray(X_ev, np.float64))).numpy())
                        z_lp[a][v] = cal_lp(np.where(np.isfinite(s), s, floor_lp))
            t_tr = time.perf_counter() - t_tr0
            ch_lh = [a for a in chans if len(z_lh[a]) == len(vids)]
            ch_lp = [a for a in chans if len(z_lp[a]) == len(vids)]
            ev_lh = evaluate(fused(z_lh, ch_lh), gt) if ch_lh else ev_cv
            ev_lp = evaluate(fused(z_lp, ch_lp), gt) if ch_lp else ev_cv
            draws_out.append({
                "seed": seed,
                "videos": len(keep),
                "scene_cv": ev_cv,
                "scene_cv_fit_s": t_cv,
                "loo_hyper": ev_lh,
                "loo_prior_blend": ev_lp,
                "loo_both_fit_s": t_tr,
            })
            print(
                f"[SHT LOO f={frac:.2f}] seed={seed}: scene_cv={ev_cv['causal']*100:.2f}% "
                f"loo_hyper={ev_lh['causal']*100:.2f}% loo_prior_blend={ev_lp['causal']*100:.2f}%",
                flush=True,
            )
        frac_results[str(frac)] = {
            "draws": draws_out,
            "scene_cv_causal_mean": float(np.mean([d["scene_cv"]["causal"] for d in draws_out])),
            "scene_cv_causal_std": float(np.std([d["scene_cv"]["causal"] for d in draws_out], ddof=1)),
            "loo_hyper_causal_mean": float(np.mean([d["loo_hyper"]["causal"] for d in draws_out])),
            "loo_hyper_causal_std": float(np.std([d["loo_hyper"]["causal"] for d in draws_out], ddof=1)),
            "loo_prior_blend_causal_mean": float(np.mean([d["loo_prior_blend"]["causal"] for d in draws_out])),
            "loo_prior_blend_causal_std": float(np.std([d["loo_prior_blend"]["causal"] for d in draws_out], ddof=1)),
        }

    return {
        "full_data": {
            "scene_cv": eval_base_full,
            "scene_cv_fit_s": full_cv_fit_s,
            "loo_hyper": eval_loo_full,
            "loo_hyper_fit_s": loo_hyper_fit_s,
            "speedup_factor": full_cv_fit_s / max(loo_hyper_fit_s, 1e-9),
            "boot_loo_vs_cv": boot_loo_full,
        },
        "fractions": frac_results,
    }


def main():
    n_threads = int(os.environ.get("TORCH_THREADS", "16"))
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    torch.set_num_threads(n_threads)
    overlap_info = sht_scene_overlap()
    print("=== Stage 3a: SHT Scene Overlap ===", flush=True)
    print(json.dumps(overlap_info, indent=2), flush=True)

    out_path = ROOT / "paper/ieee_access/kanit/birlesik_20261108/stage3_prior_dataeff.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if "f010_ablation" in existing:
            print("Loaded existing f010_ablation from", out_path, flush=True)
            f010 = existing["f010_ablation"]
        else:
            print("\n=== Stage 3b-1: f=0.10 NIW vs ML-GMM vs Plugin vs 1-NN ===", flush=True)
            f010 = run_f010_all_datasets()
    else:
        print("\n=== Stage 3b-1: f=0.10 NIW vs ML-GMM vs Plugin vs 1-NN ===", flush=True)
        f010 = run_f010_all_datasets()

    out_path.write_text(
        json.dumps(
            {
                "sht_scene_overlap": overlap_info,
                "f010_ablation": f010,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Intermediate save written to {out_path}", flush=True)

    print("\n=== Stage 3b-2: SHT Leave-One-Scene-Out Cross-Scene Prior Transfer (12 overlap scenes) ===", flush=True)
    cross_scene = run_cross_scene_prior_sht(overlap_info["overlap_scenes"])

    out_path.write_text(
        json.dumps(
            {
                "sht_scene_overlap": overlap_info,
                "f010_ablation": f010,
                "sht_cross_scene_loo": cross_scene,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote full {out_path}", flush=True)


if __name__ == "__main__":
    main()
