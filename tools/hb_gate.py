"""Normal-data gate of HB-POT stage 1 (docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md §3; procedure
in §3.4 of hiyerarsik_kalibrasyon_on_kayit_TASLAK_2026-11-05.md). Label-free: reads only the blind copies of the
normal references (tools/grid_refs_only.py; a file with any non-'ref/' array is refused) and their video ids
(tools/grid_ref_ids.py). Arms T0 (FisherCalibrator), T1 (per-scene GPD, no pooling), T2 (truncated EB mode).

LOVO, per channel, scene s and training video v: body, u_s, d_s and theta_hat_s refitted without v;
Sigma_s^(-v) = Sigma_s J_s / (J_s - 1) (Sigma_s: full-data video bootstrap, B = 500, seed 0, already x J/(J-1));
(mu, T) re-estimated by REML with scene s's estimate replaced; v's reference frames are scored. Measures:
  tail log-score  log 0.1 + log g(y) on held-out exceedances (y = x - u_s^(-v) > 0; the body terms are the same in
                  every arm); mean over the scene's exceedance frames, then unweighted mean over scenes; CI of arm
                  differences: cluster bootstrap of held-out videos within each scene, 1000 replicates, seed 0
  FAR/alpha       share of held-out frames with z > -ln alpha, over alpha (5%, 1%, 0.1%), per scene
  n(z > 10)       held-out frames, nominal probability e^-10
  consistency     aligned I + Fs + Fd sum (I frames t = 15 + 9j), pooled over scenes; per-scene share (%) above
                  the pooled 99% (and 99.9%) point; SD (ddof 0) across scenes
Gates: G1 Fd tail log-score T2 - T0 > 0 with CI lower bound > 0; G2 Fd n(z > 10) under T2 <= 12; G3 I and Fs:
n(z > 10) under T2 <= T0, and FAR/alpha at 1% in [0.5, 2.5] in >= 11 of 12 scenes under T2; G4 SD of the shares
above the pooled 99% point under T2 < T0.

    OMP_NUM_THREADS=1 python tools/hb_gate.py --refs /workspace/hb/refs/shanghaitech/{sc}_mc5.npz \
        --ids /workspace/hb/refids/shanghaitech/{sc}.npz --scenes 01 ... 12 --workers 8 --out gate.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vadkit.calibration.fisher import FisherCalibrator  # noqa: E402
from vadkit.calibration.gpd import GPDTailCalibrator  # noqa: E402
from vadkit.calibration import hier  # noqa: E402

CH = {"frx_bayes": "I", "flow_ubm_m": "Fs", "flow_rx_m": "Fd"}
ARMS = ("T0", "T1", "T2")
ALPHAS = (0.05, 0.01, 0.001)
LOG01 = math.log(0.1)
PRIOR = {"n_z10_T0": {"I": 11, "Fs": 22, "Fd": 190}, "sd99_T0_pp": 0.43}   # on_tani (diag_tail_out, tail_c_out §3)


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(refs_t: list[str], ids_t: str, scenes: list[str]) -> dict:
    """{scene: {"ref": {key}, "ids": {key}, "t": {key}, "videos"}} of the blind copies; refuses non-'ref/' arrays."""
    data = {}
    for sc in scenes:
        d = {"ref": {}, "ids": {}, "t": {}}
        for t in refs_t:
            with np.load(t.format(sc=sc)) as S:
                bad = [k for k in S.files if not k.startswith("ref/")]
                if bad:
                    raise SystemExit(f"{t.format(sc=sc)} holds non-'ref/' arrays {bad[:3]}: not a blind copy")
                for k in CH:
                    if f"ref/{k}" in S.files:
                        d["ref"][k] = S[f"ref/{k}"].astype(np.float64)
        with np.load(ids_t.format(sc=sc)) as I:
            d["videos"] = [str(v) for v in I["videos"]]
            for k in CH:
                d["ids"][k], d["t"][k] = I[f"ids/{k}"], I[f"t/{k}"]
                assert d["ids"][k].shape == d["ref"][k].shape, (sc, k)
        data[sc] = d
    return data


def boot_job(args):
    ref, vid, B, seed = args
    return hier.video_bootstrap(ref, vid, B, seed)["Sigma"]


def lovo_job(args):
    """Held-out calibrated values of every arm for one (channel, scene), video by video, and per-video tail scores."""
    ref, vid, s_idx, thetas, sigmas, J = args
    z = {a: np.empty_like(ref) for a in ARMS}
    vids = np.unique(vid)
    tail = {a: np.zeros(len(vids)) for a in ARMS}
    n_exc = np.zeros(len(vids), dtype=np.int64)
    params = []
    for i, j in enumerate(vids):
        m = vid == j
        rr, held = ref[~m], ref[m]
        c0 = FisherCalibrator().fit(rr)
        est = hier.scene_theta(rr)
        assert est["u"] == c0.u
        Sig = sigmas[s_idx] * J / (J - 1)
        star1 = hier.truncated_mode(*hier.posterior(est["theta"], Sig))
        th, sg = thetas.copy(), sigmas.copy()
        th[s_idx], sg[s_idx] = est["theta"], Sig
        mu, T = hier.eb_reml(th, sg)
        star2 = hier.truncated_mode(*hier.posterior(est["theta"], Sig, mu, T))
        cals = {"T0": GPDTailCalibrator(0.0).fit(rr),
                "T1": GPDTailCalibrator(star1[0], est["d"] * math.exp(star1[1])).fit(rr),
                "T2": GPDTailCalibrator(star2[0], est["d"] * math.exp(star2[1])).fit(rr)}
        z0 = c0(held)
        assert np.array_equal(z0, cals["T0"](held))                    # T0 = FisherCalibrator bit for bit
        z["T0"][m] = z0
        for a in ("T1", "T2"):
            z[a][m] = cals[a](held)
        y = held[held > c0.u] - c0.u
        n_exc[i] = len(y)
        for a in ARMS:
            tail[a][i] = float(np.sum(LOG01 + cals[a].tail_logpdf(y)))
        params.append({"video": int(j), "theta_hat": est["theta"].tolist(), "t1": star1.tolist(),
                       "t2": star2.tolist(), "mu": mu.tolist(), "tau": np.sqrt(np.diag(T)).tolist()})
    return z, tail, n_exc, params


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--refs", nargs="+", required=True, help="blind ref npz templates ('{sc}')")
    p.add_argument("--ids", required=True, help="ref id npz template ('{sc}'), tools/grid_ref_ids.py")
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--B", type=int, default=hier.B_BOOT)
    p.add_argument("--seed", type=int, default=hier.SEED)
    p.add_argument("--ci-reps", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    t_start = time.time()
    scenes = a.scenes
    data = load(a.refs, a.ids, scenes)
    S = len(scenes)
    print(f"loaded {S} scenes: " + ", ".join(f"{sc} J={len(data[sc]['videos'])}" for sc in scenes), flush=True)

    res = {"preregistration": "docs/tez_notlari/hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md",
           "settings": {"q": hier.Q, "xi_bounds": list(hier.XI_BOUNDS), "B": a.B, "seed": a.seed,
                        "ci_reps": a.ci_reps, "ci_seed": 0, "alphas": list(ALPHAS), "arms": list(ARMS),
                        "sigma_lovo": "Sigma_s * J_s / (J_s - 1)",
                        "tail_logscore": "mean over held-out exceedance frames of log 0.1 + log g(y), then unweighted "
                                         "mean over scenes",
                        "bootstrap_rng": "numpy default_rng(seed) per (channel, scene)"},
           "inputs": {"refs": a.refs, "ids": a.ids, "scenes": scenes,
                      "sha256": {sc: {"refs": [sha256(t.format(sc=sc)) for t in a.refs],
                                      "ids": sha256(a.ids.format(sc=sc))} for sc in scenes},
                      "code_sha256": {f: sha256(Path(__file__).resolve().parents[1] / f) for f in
                                      ("vadkit/calibration/hier.py", "vadkit/calibration/gpd.py",
                                       "vadkit/calibration/fisher.py", "tools/hb_gate.py")}},
           "channels": {}}
    Z = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, c in CH.items():
            t0 = time.time()
            full = {sc: hier.scene_theta(data[sc]["ref"][k]) for sc in scenes}
            sig = list(ex.map(boot_job, [(data[sc]["ref"][k], data[sc]["ids"][k], a.B, a.seed) for sc in scenes]))
            thetas = np.stack([full[sc]["theta"] for sc in scenes])
            sigmas = np.stack(sig)
            mu, T = hier.eb_reml(thetas, sigmas)
            ch = {"eb_full": {"mu": mu.tolist(), "tau": np.sqrt(np.diag(T)).tolist()}, "scenes": {}}
            for i, sc in enumerate(scenes):
                J = len(np.unique(data[sc]["ids"][k]))
                th1, V1 = hier.posterior(thetas[i], sigmas[i])
                th2, V2 = hier.posterior(thetas[i], sigmas[i], mu, T)
                s1, s2 = hier.truncated_mode(th1, V1), hier.truncated_mode(th2, V2)
                ch["scenes"][sc] = {"J": J, "n_ref": int(len(data[sc]["ref"][k])), "u": full[sc]["u"],
                                    "d": full[sc]["d"], "n_exc": full[sc]["n_exc"],
                                    "theta_hat": thetas[i].tolist(), "Sigma": sigmas[i].tolist(),
                                    "T1_theta_star": s1.tolist(), "T2_theta_post": th2.tolist(),
                                    "T2_V": V2.tolist(), "T2_theta_star": s2.tolist(),
                                    "T1_sigma": full[sc]["d"] * math.exp(s1[1]),
                                    "T2_sigma": full[sc]["d"] * math.exp(s2[1])}
            print(f"[{c}] full fit + bootstrap {time.time() - t0:.1f}s  mu={np.round(mu, 3).tolist()} "
                  f"tau={np.round(np.sqrt(np.diag(T)), 3).tolist()}", flush=True)
            t0 = time.time()
            jobs = [(data[sc]["ref"][k], data[sc]["ids"][k], i, thetas, sigmas,
                     len(np.unique(data[sc]["ids"][k]))) for i, sc in enumerate(scenes)]
            out = list(ex.map(lovo_job, jobs))
            print(f"[{c}] LOVO {time.time() - t0:.1f}s", flush=True)
            for sc, (z, tail, n_exc, params) in zip(scenes, out):
                Z[(c, sc)] = z
                e = ch["scenes"][sc]
                n = len(data[sc]["ref"][k])
                e["n_heldout"] = n
                e["far_over_alpha"] = {arm: {str(al): float(np.mean(z[arm] > -math.log(al)) / al) for al in ALPHAS}
                                       for arm in ARMS}
                e["n_z10"] = {arm: int(np.sum(z[arm] > 10)) for arm in ARMS}
                e["max_z"] = {arm: float(z[arm].max()) for arm in ARMS}
                e["tail"] = {"n_exc_heldout": int(n_exc.sum()), "per_video_n_exc": n_exc.tolist(),
                             "per_video_sum": {arm: tail[arm].tolist() for arm in ARMS},
                             "mean": {arm: (float(tail[arm].sum() / n_exc.sum()) if n_exc.sum() else None)
                                      for arm in ARMS}}
                e["lovo_params"] = params
            res["channels"][c] = ch

    # ---------------------------------------------------------------- summaries per channel
    summ = {}
    for c, ch in res["channels"].items():
        sc_ = ch["scenes"]
        tot = {arm: int(sum(sc_[sc]["n_z10"][arm] for sc in scenes)) for arm in ARMS}
        n_all = sum(sc_[sc]["n_heldout"] for sc in scenes)
        means = {arm: float(np.mean([sc_[sc]["tail"]["mean"][arm] for sc in scenes])) for arm in ARMS}
        # cluster bootstrap of held-out videos within each scene (same draws for every arm), 1000 reps, seed 0
        rng_ci = np.random.default_rng(0)
        per = {sc: (np.asarray(sc_[sc]["tail"]["per_video_n_exc"]),
                    {arm: np.asarray(sc_[sc]["tail"]["per_video_sum"][arm]) for arm in ARMS}) for sc in scenes}
        reps = {arm: [] for arm in ARMS}
        empty = 0
        for _ in range(a.ci_reps):
            vals = {arm: [] for arm in ARMS}
            for sc in scenes:
                cnt, sums = per[sc]
                pick = rng_ci.integers(0, len(cnt), len(cnt))
                nn = cnt[pick].sum()
                if nn == 0:
                    empty += 1
                for arm in ARMS:
                    vals[arm].append(sums[arm][pick].sum() / nn if nn else np.nan)
            for arm in ARMS:
                reps[arm].append(np.nanmean(vals[arm]))
        reps = {arm: np.asarray(v) for arm, v in reps.items()}
        ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        diffs = {}
        for a1, a2 in (("T2", "T0"), ("T1", "T0"), ("T2", "T1")):
            d = reps[a1] - reps[a2]
            diffs[f"{a1}-{a2}"] = {"point": means[a1] - means[a2], "ci95": ci(d),
                                   "fraction_positive": float(np.mean(d > 0))}
        far1 = {arm: [sc_[sc]["far_over_alpha"][arm]["0.01"] for sc in scenes] for arm in ARMS}
        summ[c] = {"n_heldout": n_all, "expected_z10": n_all * math.exp(-10), "n_z10": tot,
                   "max_z": {arm: max(sc_[sc]["max_z"][arm] for sc in scenes) for arm in ARMS},
                   "tail_logscore": {"mean": means, "differences": diffs, "scene_resamples_without_exceedance": empty},
                   "far1_in_0.5_2.5": {arm: int(sum(0.5 <= x <= 2.5 for x in far1[arm])) for arm in ARMS},
                   "far_range": {arm: {str(al): [min(sc_[sc]["far_over_alpha"][arm][str(al)] for sc in scenes),
                                                 max(sc_[sc]["far_over_alpha"][arm][str(al)] for sc in scenes)]
                                       for al in ALPHAS} for arm in ARMS}}
    res["summary"] = summ

    # ---------------------------------------------------------------- consistency of the aligned grid sum
    cons = {}
    for arm in ARMS:
        per = {}
        for sc in scenes:
            d = data[sc]
            pos = {(int(v), int(t)): i for i, (v, t) in enumerate(zip(d["ids"]["flow_ubm_m"], d["t"]["flow_ubm_m"]))}
            assert np.array_equal(d["ids"]["flow_ubm_m"], d["ids"]["flow_rx_m"])
            assert np.array_equal(d["t"]["flow_ubm_m"], d["t"]["flow_rx_m"])
            al = np.array([pos[(int(v), int(t))] for v, t in zip(d["ids"]["frx_bayes"], d["t"]["frx_bayes"])])
            per[sc] = Z[("I", sc)][arm] + Z[("Fs", sc)][arm][al] + Z[("Fd", sc)][arm][al]
        allz = np.concatenate([per[sc] for sc in scenes])
        q99, q999 = np.quantile(allz, 0.99), np.quantile(allz, 0.999)
        s99 = [100 * float(np.mean(per[sc] > q99)) for sc in scenes]
        s999 = [100 * float(np.mean(per[sc] > q999)) for sc in scenes]
        cons[arm] = {"n": int(len(allz)), "q99": float(q99), "q999": float(q999), "max": float(allz.max()),
                     "share99_pct": dict(zip(scenes, s99)), "share999_pct": dict(zip(scenes, s999)),
                     "sd99_pp": float(np.std(s99)), "sd999_pp": float(np.std(s999))}
    res["consistency"] = cons

    # ---------------------------------------------------------------- gates
    fd = summ["Fd"]
    g1 = fd["tail_logscore"]["differences"]["T2-T0"]
    g3 = {c: {"n_z10_T2": summ[c]["n_z10"]["T2"], "n_z10_T0": summ[c]["n_z10"]["T0"],
              "scenes_far1_in_range_T2": summ[c]["far1_in_0.5_2.5"]["T2"], "scenes": S,
              "pass": bool(summ[c]["n_z10"]["T2"] <= summ[c]["n_z10"]["T0"] and
                           summ[c]["far1_in_0.5_2.5"]["T2"] >= S - 1)} for c in ("I", "Fs")}
    res["gates"] = {
        "G1": {"rule": "Fd tail log-score T2 - T0 > 0 and 95% CI lower bound > 0", **g1,
               "pass": bool(g1["point"] > 0 and g1["ci95"][0] > 0)},
        "G2": {"rule": "Fd n(z > 10) under T2 <= 12", "n_z10_T2": fd["n_z10"]["T2"], "n_z10_T0": fd["n_z10"]["T0"],
               "pass": bool(fd["n_z10"]["T2"] <= 12)},
        "G3": {"rule": "I and Fs: n(z > 10) T2 <= T0 and FAR/alpha at 1% in [0.5, 2.5] in >= 11/12 scenes (T2)",
               **g3, "pass": bool(all(g3[c]["pass"] for c in g3))},
        "G4": {"rule": "SD across scenes of the share above the pooled 99% point: T2 < T0",
               "sd99_T2_pp": cons["T2"]["sd99_pp"], "sd99_T0_pp": cons["T0"]["sd99_pp"],
               "sd99_T1_pp": cons["T1"]["sd99_pp"], "pass": bool(cons["T2"]["sd99_pp"] < cons["T0"]["sd99_pp"])}}
    res["reproduction_of_prior_diagnostics"] = {
        "n_z10_T0": {c: {"now": summ[c]["n_z10"]["T0"], "prior": PRIOR["n_z10_T0"][c],
                         "match": summ[c]["n_z10"]["T0"] == PRIOR["n_z10_T0"][c]} for c in CH.values()},
        "sd99_T0_pp": {"now": cons["T0"]["sd99_pp"], "prior": PRIOR["sd99_T0_pp"],
                       "match": round(cons["T0"]["sd99_pp"], 2) == PRIOR["sd99_T0_pp"]}}
    res["elapsed_s"] = time.time() - t_start
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))

    # ---------------------------------------------------------------- printout
    for c in CH.values():
        s_ = summ[c]
        print(f"\n== {c}: held-out frames {s_['n_heldout']}, expected n(z>10) {s_['expected_z10']:.2f}")
        print("arm  n(z>10)  max z   tail log-score  FAR/a@1% in [.5,2.5]  FAR/a range 5% | 1% | 0.1%")
        for arm in ARMS:
            fr = s_["far_range"][arm]
            print(f"{arm}  {s_['n_z10'][arm]:7d} {s_['max_z'][arm]:7.2f} {s_['tail_logscore']['mean'][arm]:12.4f}"
                  f"   {s_['far1_in_0.5_2.5'][arm]:6d}/{S}            "
                  + " | ".join(f"{fr[str(al)][0]:.2f}-{fr[str(al)][1]:.2f}" for al in ALPHAS))
        for k_, d in s_["tail_logscore"]["differences"].items():
            print(f"  tail {k_}: {d['point']:+.4f} CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
        print("  per scene (J, xi_hat, T1 xi*, T2 xi*, FAR/a@1% T0/T1/T2, n(z>10) T0/T1/T2):")
        for sc in scenes:
            e = res["channels"][c]["scenes"][sc]
            f1 = [e["far_over_alpha"][arm]["0.01"] for arm in ARMS]
            print(f"   {sc} J={e['J']:3d} xi_hat {e['theta_hat'][0]:+.3f} T1 {e['T1_theta_star'][0]:.3f} "
                  f"T2 {e['T2_theta_star'][0]:.3f} | " + " ".join(f"{x:.2f}" for x in f1) + " | "
                  + " ".join(str(e["n_z10"][arm]) for arm in ARMS))
    print("\n== consistency (aligned I+Fs+Fd, share above pooled q99, %):")
    for arm in ARMS:
        cc = cons[arm]
        print(f"{arm}: q99 {cc['q99']:.2f} q99.9 {cc['q999']:.2f} max {cc['max']:.2f} sd99 {cc['sd99_pp']:.3f} "
              f"sd99.9 {cc['sd999_pp']:.3f} | " + " ".join(f"{x:.1f}" for x in cc["share99_pct"].values()))
    print("\n== gates")
    for g, v in res["gates"].items():
        print(f"{g}: {'PASS' if v['pass'] else 'FAIL'}  ({v['rule']})")
    print("\n== reproduction of prior diagnostics (T0): " + json.dumps(res["reproduction_of_prior_diagnostics"]))
    print(f"\nwrote {a.out} ({res['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
