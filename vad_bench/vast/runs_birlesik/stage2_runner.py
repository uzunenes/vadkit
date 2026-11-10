#!/usr/bin/env python3
"""Stage 2 Unified Reference Host Benchmark & Setup Measurement Runner.
Runs on vast.ai instance (AMD EPYC 7452 32-Core + NVIDIA GeForce RTX 3090).

Executes:
1. Environment & Host hardware probe
2. Data preparation (UCSD Ped2, Avenue cache and frames)
3. TensorRT engine export timing (YOLO11n + YOLO11n-pose)
4. Feature extraction setup timings (PyTorch sequential vs TensorRT) on Ped2 & Avenue
5. Object branch fitting setup timings (GPU vs CPU) on Ped2 & Avenue
6. Object branch evaluation (causal micro/macro AUC consistency)
7. Grid branch fitting and checkpoint check (Ped2 & Avenue)
8. Stream evaluation on Avenue test videos
9. End-to-end streaming latency & speed benchmark (ABBAAB sequence, 24 runs)
10. Generates stage2_timings.json consolidating all measured setup and benchmark numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def run_cmd(cmd: list[str] | str, log_file: Path | None = None, check: bool = True) -> tuple[int, float]:
    """Execute command, optionally redirect output to log_file, return (exit_code, elapsed_sec)."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    log(f"RUN: {cmd_str}")
    t0 = time.perf_counter()
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            proc = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, shell=isinstance(cmd, str), text=True)
    elapsed = time.perf_counter() - t0
    log(f"EXIT {proc.returncode} in {elapsed:.2f}s")
    if check and proc.returncode != 0:
        if log_file and log_file.exists():
            log(f"--- LOG TAIL ({log_file.name}) ---")
            lines = log_file.read_text().splitlines()
            for l in lines[-25:]:
                log(f"  {l}")
        raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd_str}")
    return proc.returncode, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace")
    args = ap.parse_args()

    ws = Path(args.workspace)
    cctv = ws / "cctv_aean"
    logs = ws / "logs"
    out = ws / "out"
    ck = ws / "checkpoints"
    data = ws / "data"
    cache = ws / "cache"
    frames = ws / "frames"

    for d in [logs, out / "feat", out / "state", out / "grid", out / "bench", out / "stream"]:
        d.mkdir(parents=True, exist_ok=True)

    os.chdir(cctv)
    os.environ["PYTHONPATH"] = str(cctv)

    timings: dict[str, any] = {"host": {}, "prep": {}, "export_trt": {}, "extract": {}, "fit": {}, "grid": {}, "stream": {}, "bench": {}}

    # -------------------------------------------------------------
    # 1. Environment & Hardware Probe
    # -------------------------------------------------------------
    log("=== Step 1: Probing Environment & Hardware ===")
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        cpu_model = platform.processor()

    gpu_name = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import ultralytics, tensorrt, cv2
        versions = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu_name,
            "cpu": cpu_model,
            "cv2": cv2.__version__,
            "ultralytics": ultralytics.__version__,
            "tensorrt": tensorrt.__version__,
            "python": platform.python_version(),
            "nproc": str(os.cpu_count()),
        }
    except Exception as e:
        log(f"Warning probing packages: {e}")
        versions = {}

    timings["host"] = versions
    (logs / "env.json").write_text(json.dumps(versions, indent=2))
    run_cmd(f"pip freeze > {logs}/pip_freeze.txt", check=False)
    log(f"Host: CPU={cpu_model}, GPU={gpu_name}, Torch={versions.get('torch')}, TRT={versions.get('tensorrt')}")

    # -------------------------------------------------------------
    # 2. Preparation (Cache and Frames)
    # -------------------------------------------------------------
    log("=== Step 2: Preparing Cache and Frames ===")
    # Unpack ucsd if needed
    if not (data / "UCSD_Anomaly_Dataset.v1p2").is_dir():
        run_cmd(f"tar xzf {data}/UCSD_Anomaly_Dataset.tar.gz -C {data}")
    # Prepare avenue directory if needed
    (data / "avenue").mkdir(parents=True, exist_ok=True)
    if (data / "Avenue_Dataset.zip").exists() and not (data / "avenue/Avenue_Dataset.zip").exists():
        shutil.move(str(data / "Avenue_Dataset.zip"), str(data / "avenue/Avenue_Dataset.zip"))
    if (data / "ground_truth_demo.zip").exists() and not (data / "avenue/ground_truth_demo.zip").exists():
        shutil.move(str(data / "ground_truth_demo.zip"), str(data / "avenue/ground_truth_demo.zip"))

    # Unpack checkpoints if needed
    if not ck.is_dir():
        run_cmd(f"tar xf {data}/checkpoints_2026-11-02.tar -C {ws}")

    # Run prep scripts
    run_cmd(f"python prepare_ucsd.py --root {data}/UCSD_Anomaly_Dataset.v1p2 --out {cache} --peds UCSDped2",
            logs / "prep_ucsd.log")
    run_cmd(f"python prepare_avenue.py --zip {data}/avenue/Avenue_Dataset.zip --gt-zip {data}/avenue/ground_truth_demo.zip --out {cache}/avenue --splits train,test",
            logs / "prep_avenue.log")

    # Hazirla frames (parallel)
    log("Extracting JPEG frames for Ped2 and Avenue...")
    p1 = subprocess.Popen(f"python vad_bench/vast/hazirla_frames.py --ds ped2 --src {data}/UCSD_Anomaly_Dataset.v1p2 --out {frames} --gt {cache}/ped2/gt.json > {logs}/frames_ped2.log 2>&1", shell=True)
    p2 = subprocess.Popen(f"python vad_bench/vast/hazirla_frames.py --ds avenue --src {data}/avenue/Avenue_Dataset.zip --out {frames} --gt {cache}/avenue/gt.json > {logs}/frames_avenue.log 2>&1", shell=True)
    p1.wait(); p2.wait()
    assert p1.returncode == 0 and p2.returncode == 0, "hazirla_frames failed"
    log("Frames preparation completed.")

    # -------------------------------------------------------------
    # 3. Export TensorRT Engines (Timed in isolated subprocesses)
    # -------------------------------------------------------------
    log("=== Step 3: Exporting TensorRT Engines (Timed) ===")
    _, t_yolo = run_cmd(
        'python -c "from ultralytics import YOLO; YOLO(\'yolo11n.pt\').export(format=\'engine\', half=True, dynamic=True)"',
        logs / "export_yolo11n.log"
    )
    _, t_pose = run_cmd(
        'python -c "from ultralytics import YOLO; YOLO(\'yolo11n-pose.pt\').export(format=\'engine\', half=True, dynamic=True)"',
        logs / "export_yolo11n_pose.log"
    )

    timings["export_trt"] = {
        "yolo11n_engine_s": round(t_yolo, 2),
        "yolo11n_pose_engine_s": round(t_pose, 2),
        "total_trt_build_s": round(t_yolo + t_pose, 2),
        "total_trt_build_min": round((t_yolo + t_pose) / 60.0, 2),
    }
    log(f"TRT Engine build: YOLO11n={t_yolo:.1f}s, YOLO11n-pose={t_pose:.1f}s (Total: {(t_yolo+t_pose)/60:.2f} min)")

    # -------------------------------------------------------------
    # 4. Feature Extraction Setup Timings (PyTorch vs TensorRT)
    # -------------------------------------------------------------
    log("=== Step 4: Feature Extraction Setup Timings ===")
    # (a) PyTorch sequential extraction on Ped2 (train + test)
    log("Measuring Ped2 PyTorch feature extraction (training)...")
    _, t_ped2_pt_train = run_cmd(
        f"python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames {frames}/ped2 --dataset ped2 --splits training --out {out}/feat/ped2 --device cuda",
        logs / "extract_ped2_train_pt.log"
    )
    log("Extracting Ped2 PyTorch feature extraction (testing)...")
    _, t_ped2_pt_test = run_cmd(
        f"python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames {frames}/ped2 --dataset ped2 --splits testing --out {out}/feat/ped2 --device cuda",
        logs / "extract_ped2_test_pt.log"
    )

    # (b) PyTorch sequential extraction on Avenue (train + test)
    log("Measuring Avenue PyTorch feature extraction (training)...")
    _, t_av_pt_train = run_cmd(
        f"python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames {frames}/avenue --dataset avenue --splits training --out {out}/feat/avenue --device cuda",
        logs / "extract_avenue_train_pt.log"
    )
    log("Extracting Avenue PyTorch feature extraction (testing)...")
    _, t_av_pt_test = run_cmd(
        f"python -m vadkit extract --config vadkit/configs/r2/base.yaml --frames {frames}/avenue --dataset avenue --splits testing --out {out}/feat/avenue --device cuda",
        logs / "extract_avenue_test_pt.log"
    )

    # (c) TensorRT + parallel flow feature extraction on Ped2 (training)
    log("Measuring Ped2 TensorRT + parallel flow feature extraction (training)...")
    _, t_ped2_trt_train = run_cmd(
        f"python -m vadkit extract --config vadkit/configs/r2/base_trt_par.yaml --frames {frames}/ped2 --dataset ped2 --splits training --out {out}/feat/ped2_trt --device cuda",
        logs / "extract_ped2_train_trt.log"
    )

    # (d) TensorRT + parallel flow feature extraction on Avenue (training)
    log("Measuring Avenue TensorRT + parallel flow feature extraction (training)...")
    _, t_av_trt_train = run_cmd(
        f"python -m vadkit extract --config vadkit/configs/r2/base_trt_par.yaml --frames {frames}/avenue --dataset avenue --splits training --out {out}/feat/avenue_trt --device cuda",
        logs / "extract_avenue_train_trt.log"
    )

    timings["extract"] = {
        "ped2_pt_train_s": round(t_ped2_pt_train, 2),
        "ped2_pt_train_min": round(t_ped2_pt_train / 60.0, 2),
        "ped2_pt_test_s": round(t_ped2_pt_test, 2),
        "avenue_pt_train_s": round(t_av_pt_train, 2),
        "avenue_pt_train_min": round(t_av_pt_train / 60.0, 2),
        "avenue_pt_test_s": round(t_av_pt_test, 2),
        "ped2_trt_train_s": round(t_ped2_trt_train, 2),
        "ped2_trt_train_min": round(t_ped2_trt_train / 60.0, 2),
        "avenue_trt_train_s": round(t_av_trt_train, 2),
        "avenue_trt_train_min": round(t_av_trt_train / 60.0, 2),
    }
    log(f"Extraction summary: Ped2 PyTorch train={t_ped2_pt_train/60:.2f} min, TRT train={t_ped2_trt_train/60:.2f} min")
    log(f"Extraction summary: Avenue PyTorch train={t_av_pt_train/60:.2f} min, TRT train={t_av_trt_train/60:.2f} min")

    # -------------------------------------------------------------
    # 5. Object Branch Fitting Setup Timings (GPU vs CPU)
    # -------------------------------------------------------------
    log("=== Step 5: Object Branch Fitting Setup Timings ===")
    # Ped2 GPU fit
    log("Timing Ped2 GPU fit...")
    _, t_ped2_fit_gpu = run_cmd(
        f"python -m vadkit fit --config vadkit/configs/r2/base.yaml --features {out}/feat/ped2 --out {out}/state/ped2 --device cuda",
        logs / "fit_ped2_gpu.log"
    )
    # Ped2 CPU fit
    log("Timing Ped2 CPU fit (AMD EPYC 7452)...")
    _, t_ped2_fit_cpu = run_cmd(
        f"python -m vadkit fit --config vadkit/configs/r2/base.yaml --features {out}/feat/ped2 --out {out}/state/ped2_cpu --device cpu",
        logs / "fit_ped2_cpu.log"
    )

    # Avenue GPU fit
    log("Timing Avenue GPU fit...")
    _, t_av_fit_gpu = run_cmd(
        f"python -m vadkit fit --config vadkit/configs/r2/base.yaml --features {out}/feat/avenue --out {out}/state/avenue --device cuda",
        logs / "fit_avenue_gpu.log"
    )
    # Avenue CPU fit
    log("Timing Avenue CPU fit (AMD EPYC 7452)...")
    _, t_av_fit_cpu = run_cmd(
        f"python -m vadkit fit --config vadkit/configs/r2/base.yaml --features {out}/feat/avenue --out {out}/state/avenue_cpu --device cpu",
        logs / "fit_avenue_cpu.log"
    )

    timings["fit"] = {
        "ped2_gpu_s": round(t_ped2_fit_gpu, 2),
        "ped2_cpu_s": round(t_ped2_fit_cpu, 2),
        "ped2_cpu_min": round(t_ped2_fit_cpu / 60.0, 2),
        "avenue_gpu_s": round(t_av_fit_gpu, 2),
        "avenue_cpu_s": round(t_av_fit_cpu, 2),
        "avenue_cpu_min": round(t_av_fit_cpu / 60.0, 2),
    }
    log(f"Fitting summary: Ped2 GPU={t_ped2_fit_gpu:.2f}s, CPU={t_ped2_fit_cpu:.2f}s ({t_ped2_fit_cpu/60:.2f} min)")
    log(f"Fitting summary: Avenue GPU={t_av_fit_gpu:.2f}s, CPU={t_av_fit_cpu:.2f}s ({t_av_fit_cpu/60:.2f} min)")

    # -------------------------------------------------------------
    # 6. Object Branch Evaluation
    # -------------------------------------------------------------
    log("=== Step 6: Object Branch Evaluation ===")
    run_cmd(
        f"python -m vadkit eval --config vadkit/configs/r2/base.yaml --features {out}/feat/ped2 --state {out}/state/ped2 --gt {cache}/ped2/gt.json --out {out}/state/eval_obj_ped2.json",
        logs / "eval_obj_ped2.log"
    )
    run_cmd(
        f"python -m vadkit eval --config vadkit/configs/r2/base.yaml --features {out}/feat/avenue --state {out}/state/avenue --gt {cache}/avenue/gt.json --out {out}/state/eval_obj_avenue.json",
        logs / "eval_obj_avenue.log"
    )
    log("Object evaluation completed.")

    # -------------------------------------------------------------
    # 7. Grid Branch Fitting & Check
    # -------------------------------------------------------------
    log("=== Step 7: Grid Branch Fitting & Check ===")
    for ds in ["ped2", "avenue"]:
        run_cmd(
            f"python tools/grid_check.py --cache {cache}/{ds} --dataset {ds} --run {ck}/runs/{ds}_2d_noadv --archive {ck}/scores/runs/{ds}_2d_noadv --state {out}/grid/state_{ds} --out {out}/grid/check_{ds}.json --device cuda --workers 8",
            logs / f"grid_{ds}.log"
        )
    log("Grid check completed.")

    # -------------------------------------------------------------
    # 8. Stream Eval on Avenue
    # -------------------------------------------------------------
    log("=== Step 8: Streaming Combined Detector Evaluation on Avenue ===")
    run_cmd(
        f"python tools/stream_eval.py --config vadkit/configs/r2/base.yaml --frames {frames}/avenue --dataset avenue --state {out}/state/avenue --grid-state {out}/grid/state_avenue --grid-run {ck}/runs/avenue_2d_noadv --gt {cache}/avenue/gt.json --out {out}/stream/avenue",
        logs / "stream_avenue.log"
    )
    log("Stream evaluation completed.")

    # -------------------------------------------------------------
    # 9. End-to-End Latency & Speed Benchmark (ABBAAB sequence, 24 runs)
    # -------------------------------------------------------------
    log("=== Step 9: End-to-End Latency & Speed Benchmark (24 runs) ===")
    run_cmd("nvidia-smi > " + str(out / "bench/nvidia_smi_start.txt") + " 2>&1; uptime >> " + str(out / "bench/nvidia_smi_start.txt"))

    cfg_map = {"trt": "vadkit/configs/r2/base_trt.yaml", "pt": "vadkit/configs/r2/base.yaml"}
    modes = ["obj", "grid", "grid", "obj", "obj", "grid"]

    for ds in ["avenue", "ped2"]:
        for be in ["trt", "pt"]:
            cfg_file = cfg_map[be]
            for i, mode in enumerate(modes, 1):
                out_bench = out / f"bench/{ds}_{be}_{mode}_{i}.json"
                log_bench = logs / f"bench_{ds}_{be}_{mode}_{i}.log"
                grid_flag = f"--grid-state {out}/grid/state_{ds} --grid-run {ck}/runs/{ds}_2d_noadv --parallel-grid" if mode == "grid" else ""
                log(f"Benchmark [{ds} {be} {mode} run {i}/6]...")
                run_cmd(
                    f"python tools/bench_e2e.py --config {cfg_file} --frames {frames}/{ds} --dataset {ds} --state {out}/state/{ds} --parallel-flow {grid_flag} --out {out_bench}",
                    log_bench
                )

    run_cmd("nvidia-smi > " + str(out / "bench/nvidia_smi_end.txt") + " 2>&1; uptime >> " + str(out / "bench/nvidia_smi_end.txt"))
    run_cmd(f"cp {logs}/*.log {out}/ 2>/dev/null", check=False)

    # Save overall timings
    (out / "stage2_timings.json").write_text(json.dumps(timings, indent=2))
    (logs / "ALL_DONE").touch()
    log("=== ALL STAGE 2 STEPS COMPLETED SUCCESSFULLY! ===")


if __name__ == "__main__":
    main()
