#!/usr/bin/env python3
"""
benchmark_baseline_gpu.py - FP32 reference numbers on the desktop GPU.

    checkpoint/pytorch/<task>/<model>.pt     -> PyTorch FP32 accuracy + latency
    checkpoint/onnx/<task>/<model>.onnx      -> ONNX Runtime FP32 accuracy (export parity check)
        -> data/benchmark/<task>/baseline_gpu.csv

WHY A SEPARATE BASELINE, WHEN cnn_training.csv ALREADY HAS top1
---------------------------------------------------------------
The accuracy recorded during training was measured inside the training loop under bfloat16
autocast. It is the right number for "did this model train", but it is the WRONG number to
subtract an INT8 result from, because part of any gap would be the harness rather than the
quantisation. A quantisation-degradation claim is only valid if the FP32 and INT8 numbers come
out of the SAME evaluation procedure: same images, same preprocessing, same top-k arithmetic.
This script is that procedure, and the per-backend benchmark scripts will reuse it.

TWO ARTEFACTS PER MODEL, ON PURPOSE
-----------------------------------
  pytorch-fp32  the true reference: the trained weights, nothing converted. Full val set.
  onnx-fp32     the graph every quantised binary descends from. A SUBSET is enough - see below.

Comparing the two closes a gap in the pipeline. export_onnx.py already verified PyTorch vs ONNX
parity, but on RANDOM inputs. Random tensors can hide a defect that only shows on real images -
a layout error, a wrongly-folded BatchNorm, a frozen branch. If these two accuracies differ by
more than sampling noise, the export is broken and every INT8 number built on it is void.

The ONNX pass runs on a SUBSET (default 1000 images) because the exported graphs have a static
batch of 1, so a full 5000-image pass is 5000 sequential forward passes per model. The pass
exists to DETECT a broken export, and a broken export shifts accuracy by whole percentage
points, not tenths - 1000 images is ample. Use --onnx-eval-n 0 for the full set.

WHAT THE LATENCY NUMBERS ARE AND ARE NOT
----------------------------------------
A desktop GPU is not an edge device, so these latencies do NOT belong in a like-for-like table
with Jetson, Coral, Hailo or the Pi. They are useful for three other things:
  1. An unconstrained upper bound: what the architecture costs when memory bandwidth and power
     are not the binding constraint.
  2. A fifth ranking, which lets the Kendall-tau machinery be written and sanity-checked now,
     before any accelerator arrives.
  3. A first test of "do MACs predict latency", on hardware you already have.

MEASUREMENT HYGIENE (why the timing loop looks the way it does)
---------------------------------------------------------------
  * Warmup iterations are discarded: the first calls pay cuDNN autotuning, memory allocation and
    kernel JIT, which are one-off costs that would otherwise dominate a short run.
  * torch.cuda.synchronize() brackets every timed region. CUDA launches are asynchronous, so
    without it the clock measures how fast Python QUEUED the work, not how long the GPU took.
  * p50/p95/p99 are reported, not just the mean. A control loop cares about the tail.
  * Clocks drift on a consumer card under sustained load. Fix them before a run that will be
    quoted, and record the GPU name either way.

Usage
-----
    python scripts/benchmark_baseline_gpu.py --task classification
    python scripts/benchmark_baseline_gpu.py --task classification --only mobilenetv2_100
    python scripts/benchmark_baseline_gpu.py --task classification --skip-onnx
    python scripts/benchmark_baseline_gpu.py --task classification --quick   # harness smoke test

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
ROOT = SCRIPT_DIR.parent

# Reuse the checkpoint-rebuilding logic rather than duplicating it: if build_model() ever gains
# a model registry for the external-repo architectures, this script inherits it for free.
from export_onnx import load_checkpoint, rebuild_from_checkpoint   # noqa: E402

PT_ROOT = ROOT / "checkpoint" / "pytorch"
ONNX_ROOT = ROOT / "checkpoint" / "onnx"
OUT_ROOT = ROOT / "data" / "benchmark"
CACHE_DIR = ROOT / "datasets" / "imageNet-100_cache"

# Same constants as training and as edge_common. Kept in float [0,1] space here because the
# evaluation path works in floats throughout.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def enable_windows_cuda_dlls() -> None:
    """Let onnxruntime-gpu find CUDA DLLs on Windows via PyTorch's bundled copies.

    onnxruntime-gpu links against cublasLt/cudnn DLLs that are NOT on PATH unless a full CUDA
    toolkit is installed. PyTorch ships its own, so pointing the loader at those avoids a second
    toolkit install. Harmless if the DLLs are absent or the versions do not match - ORT then
    falls back to CPU, which the provider reporting below makes visible rather than silent.
    """
    if sys.platform != "win32":
        return
    roots = [Path(torch.__file__).parent / "lib", Path(torch.__file__).parent.parent / "nvidia"]
    for r in roots:
        if not r.is_dir():
            continue
        dirs = [r] if r.name == "lib" else [d for d in r.rglob("bin") if d.is_dir()]
        for d in dirs:
            try:
                os.add_dll_directory(str(d))
            except (OSError, AttributeError):
                pass


# ---------------------------------------------------------------------------------------
# Validation data
# ---------------------------------------------------------------------------------------
_VAL_CACHE: dict[int, tuple] = {}


def load_val(res: int):
    """Load the validation cache for ONE resolution: uint8 NHWC images + int64 labels.

    Memoised per resolution. Models in a sweep may have different eval resolutions, and each
    MUST be evaluated against images at its own size - scoring a 160px model on 224px images
    would produce a wrong but entirely plausible-looking accuracy, which is the worst kind of bug.
    """
    if res in _VAL_CACHE:
        return _VAL_CACHE[res]
    fx, fy = CACHE_DIR / f"val_{res}_x.npy", CACHE_DIR / f"val_{res}_y.npy"
    if not (fx.exists() and fy.exists()):
        avail = sorted(p.name for p in CACHE_DIR.glob("val_*_x.npy")) if CACHE_DIR.is_dir() else []
        raise FileNotFoundError(
            f"no val cache at {fx}\n  available: {avail}\n"
            f"  Build it in the training notebook: build_cache('val', {res}, round({res}/0.875))")
    x, y = np.load(fx, mmap_mode="r"), np.load(fy)
    assert x.shape[0] == y.shape[0], f"val cache mismatch: {x.shape[0]} images vs {y.shape[0]} labels"
    assert x.shape[1] == x.shape[2] == res, f"val cache is {x.shape[1]}px, expected {res}px"
    _VAL_CACHE[res] = (x, y)
    return x, y


def batches(x_mm, y, bs: int, limit: int = 0):
    """Yield (uint8 NHWC batch, int64 labels). Sequential, so memmap reads stay in order."""
    n = min(limit, len(y)) if limit else len(y)
    for i in range(0, n, bs):
        # min(i+bs, n), NOT i+bs: the last batch must be truncated to the limit, or a limit of
        # 100 with batch 64 would quietly return 128 images and the subset comparison would be
        # over a different sample than intended.
        j = min(i + bs, n)
        # np.array() forces a writable copy. A memmap slice is read-only, and torch.from_numpy
        # on a read-only buffer warns and yields a tensor unsafe to write in place.
        yield np.array(x_mm[i:j]), y[i:j]


def to_nchw_float(batch_u8: np.ndarray, device) -> torch.Tensor:
    """uint8 NHWC [0,255] -> normalised float32 NCHW on the device."""
    t = torch.from_numpy(batch_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    return (t - MEAN.to(device)) / STD.to(device)


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def topk_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Return (top-1 hits, top-5 hits) for one batch. Identical arithmetic to the notebook."""
    k = min(5, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    hit = topk == labels[:, None]
    return int(hit[:, 0].sum()), int(hit.any(dim=1).sum())


def latency_stats(times_ms: list[float]) -> dict:
    """Percentiles, not just the mean - the tail is what a real-time loop has to budget for."""
    a = np.asarray(times_ms)
    return dict(lat_mean_ms=round(float(a.mean()), 4),
                lat_p50_ms=round(float(np.percentile(a, 50)), 4),
                lat_p95_ms=round(float(np.percentile(a, 95)), 4),
                lat_p99_ms=round(float(np.percentile(a, 99)), 4),
                lat_min_ms=round(float(a.min()), 4),
                lat_std_ms=round(float(a.std()), 4),
                lat_iters=len(a))


# ---------------------------------------------------------------------------------------
# PyTorch path
# ---------------------------------------------------------------------------------------
@torch.inference_mode()
def eval_pytorch(model, x_mm, y, device, bs: int, limit: int = 0) -> dict:
    """FP32 accuracy - no autocast, unlike the training loop. limit=0 means the full val set."""
    model.eval()
    c1 = c5 = n = 0
    t0 = time.time()
    for xb, yb in batches(x_mm, y, bs, limit=limit):
        logits = model(to_nchw_float(xb, device))
        a, b = topk_counts(logits.float(), torch.from_numpy(yb).to(device))
        c1 += a
        c5 += b
        n += len(yb)
    if device == "cuda":
        torch.cuda.synchronize()
    # n is COUNTED, never assumed. Dividing by an assumed total would silently inflate or
    # deflate accuracy if the loop ever yielded fewer images than expected.
    expected = min(limit, len(y)) if limit else len(y)
    assert n == expected, f"evaluated {n} images, expected {expected}"
    return dict(top1=round(100 * c1 / n, 3), top5=round(100 * c5 / n, 3),
                n_eval=n, eval_s=round(time.time() - t0, 1))


@torch.inference_mode()
def latency_pytorch(model, res: int, device, warmup: int, iters: int) -> dict:
    """Batch-1 latency. Each iteration is timed individually so percentiles are meaningful."""
    model.eval()
    g = torch.Generator().manual_seed(0)       # identical input for every model and every run
    x = torch.randn(1, 3, res, res, generator=g).to(device)

    for _ in range(warmup):            # absorb autotuning, allocation and JIT
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device == "cuda":
            torch.cuda.synchronize()   # ensure the PREVIOUS iteration has fully retired
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()   # ensure THIS iteration has fully retired before stopping
        times.append((time.perf_counter() - t0) * 1000)
    return latency_stats(times)


@torch.inference_mode()
def throughput_pytorch(model, res: int, device, bs: int, warmup: int, iters: int) -> dict:
    """Images/second at a larger batch - a different question from batch-1 latency.

    Batch 1 measures how fast one inference completes; throughput measures how well the
    architecture fills the device. Depthwise-heavy models usually look far worse here, because
    they are memory-bound and a bigger batch does not help them.

    Halves the batch on OOM rather than failing the model: a throughput figure at a smaller
    batch is still useful, and the batch ACTUALLY used is recorded beside it so the numbers
    are never silently incomparable.
    """
    model.eval()
    while bs >= 1:
        try:
            x = torch.randn(bs, 3, res, res, device=device)
            for _ in range(warmup):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            return dict(throughput_bs=bs, throughput_img_s=round(bs * iters / dt, 1))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            bs //= 2
            print(f"      (throughput OOM, retrying at batch {bs})")
    return dict(throughput_bs=None, throughput_img_s=None)


# ---------------------------------------------------------------------------------------
# ONNX Runtime path
# ---------------------------------------------------------------------------------------
def eval_onnx(onnx_path: Path, x_mm, y, provider: str, res: int,
              limit: int, lat_iters: int) -> dict:
    """Accuracy + latency through ONNX Runtime, to verify the exported graph on REAL images."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), so, providers=[provider])

    # ORT SILENTLY FALLS BACK TO CPU when a provider cannot be created (missing CUDA DLLs, wrong
    # cuDNN version, unsupported GPU). Reporting the REQUESTED provider instead of the actual one
    # would record a CPU latency as a GPU latency. Always report what was really used.
    actual = sess.get_providers()[0]

    inp = sess.get_inputs()[0]
    name = inp.name
    # The graph's own input shape is authoritative. If it disagrees with the val images we are
    # about to feed it, stop: ORT would either raise something cryptic or, worse, accept it.
    g_res = inp.shape[2] if isinstance(inp.shape[2], int) else None
    if g_res is not None and g_res != res:
        raise ValueError(f"{onnx_path.name} expects {g_res}px but the val images are {res}px")
    # Exported graphs are static batch 1, so evaluation runs one image at a time.
    step = 1 if inp.shape[0] in (1, "1") else 32

    c1 = c5 = n = 0
    t0 = time.time()
    for xb, yb in batches(x_mm, y, step, limit=limit):
        xt = to_nchw_float(xb, "cpu").numpy()
        logits = torch.from_numpy(sess.run(None, {name: xt})[0])
        a, b = topk_counts(logits, torch.from_numpy(yb))
        c1 += a
        c5 += b
        n += len(yb)

    # Latency through the SAME session, so it is directly comparable to the accuracy run above.
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((1, 3, res, res), dtype=np.float32)
    for _ in range(min(10, lat_iters)):
        sess.run(None, {name: x1})
    times = []
    for _ in range(lat_iters):
        t = time.perf_counter()
        sess.run(None, {name: x1})
        times.append((time.perf_counter() - t) * 1000)

    return dict(top1=round(100 * c1 / n, 3), top5=round(100 * c5 / n, 3), n_eval=n,
                eval_s=round(time.time() - t0, 1),
                provider=actual, provider_requested=provider,
                provider_fell_back=(actual != provider),
                ort_version=ort.__version__, **latency_stats(times))


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------
def run_model(pt_path: Path, task: str, args, device) -> list[dict]:
    """Benchmark one checkpoint: the PyTorch artefact, then its ONNX counterpart."""
    stem = pt_path.stem
    rows = []

    ck = load_checkpoint(pt_path)
    model, meta = rebuild_from_checkpoint(ck, fallback_name=stem)
    # PER-MODEL resolution, with the matching val cache. Models in one sweep may differ, and
    # each must be scored at its own size.
    res = args.res or int(meta.get("eval_res") or 224)
    x_mm, y = load_val(res)
    model = model.to(device)

    common = dict(model=stem, task=task, res=res, precision="fp32",
                  source=meta.get("source"), recipe_hash=meta.get("recipe_hash"),
                  train_top1=meta.get("top1"),   # the autocast number, for comparison only
                  device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                  torch_version=torch.__version__)

    # ---- PyTorch ----
    r = dict(common, artifact="pytorch-fp32", status="ok", n_val_total=len(y))
    r.update(eval_pytorch(model, x_mm, y, device, args.batch))
    # Reset AFTER the accuracy pass so peak VRAM reflects batch-1 inference, which is the
    # deployment-relevant figure. The accuracy pass uses a large batch and would dominate it.
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    r.update(latency_pytorch(model, res, device, args.warmup, args.iters))
    if device == "cuda":
        r["peak_vram_bs1_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    r.update(throughput_pytorch(model, res, device, args.throughput_batch, 3, args.throughput_iters))
    rows.append(r)
    print(f"      pytorch-fp32  top1 {r['top1']:.2f}  top5 {r['top5']:.2f}  "
          f"p50 {r['lat_p50_ms']:.2f}ms  {r['throughput_img_s']} img/s")

    # ---- ONNX Runtime ----
    onnx_path = ONNX_ROOT / task / f"{stem}.onnx"
    if args.skip_onnx:
        pass
    elif not onnx_path.exists():
        print(f"      onnx-fp32     skipped: {onnx_path.name} not found")
    else:
        ro = dict(common, artifact="onnx-fp32", status="ok")
        ro.update(eval_onnx(onnx_path, x_mm, y, args.onnx_provider, res,
                            args.onnx_eval_n, args.onnx_lat_iters))
        if ro["provider_fell_back"]:
            print(f"      NOTE: requested {ro['provider_requested']}, ORT actually used "
                  f"{ro['provider']} - this latency is NOT a GPU number")

        # THE CHECK THIS SCRIPT EXISTS FOR: the exported graph must agree with the source
        # weights on REAL images, not just the random tensors export_onnx.py used.
        # Compare like with like: if the ONNX pass used a subset, re-score PyTorch on the SAME
        # subset, otherwise part of the difference is merely a different sample of images.
        if ro["n_eval"] == r["n_eval"]:
            pt_top1 = r["top1"]
        else:
            pt_top1 = eval_pytorch(model, x_mm, y, device, args.batch, limit=ro["n_eval"])["top1"]
            ro["pytorch_top1_same_subset"] = pt_top1
        ro["top1_delta_vs_pytorch"] = round(ro["top1"] - pt_top1, 3)
        if abs(ro["top1_delta_vs_pytorch"]) > args.export_tolerance:
            ro["warning"] = (f"ONNX top-1 differs from PyTorch by "
                             f"{ro['top1_delta_vs_pytorch']:+.2f} pp on the same {ro['n_eval']} "
                             f"images - the export is suspect, and every INT8 binary derived "
                             f"from it is void")
            print(f"      *** {ro['warning']}")
        rows.append(ro)
        print(f"      onnx-fp32     top1 {ro['top1']:.2f} on {ro['n_eval']} imgs  "
              f"(delta {ro['top1_delta_vs_pytorch']:+.2f} pp)  "
              f"p50 {ro['lat_p50_ms']:.2f}ms  [{ro['provider']}]")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="classification")
    ap.add_argument("--only", nargs="*", default=None, metavar="MODEL")
    ap.add_argument("--res", type=int, default=None,
                    help="override eval resolution for ALL models (default: each checkpoint's own)")
    ap.add_argument("--batch", type=int, default=128, help="accuracy-pass batch size")
    ap.add_argument("--warmup", type=int, default=50, help="discarded latency iterations")
    ap.add_argument("--iters", type=int, default=500, help="timed latency iterations")
    ap.add_argument("--throughput-batch", type=int, default=64)
    ap.add_argument("--throughput-iters", type=int, default=30)
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--onnx-eval-n", type=int, default=1000,
                    help="images for the ONNX parity check (0 = full val set). The check detects "
                         "a BROKEN export, which shifts accuracy by whole points, so a subset "
                         "suffices - and the graphs are static batch 1, so a full pass is slow.")
    ap.add_argument("--onnx-lat-iters", type=int, default=100)
    ap.add_argument("--onnx-provider", default="CPUExecutionProvider",
                    help="CUDAExecutionProvider needs onnxruntime-gpu AND matching CUDA/cuDNN; "
                         "ORT falls back to CPU if unavailable, which is REPORTED, not hidden")
    ap.add_argument("--export-tolerance", type=float, default=0.5,
                    help="max acceptable top-1 gap (pp) between PyTorch and ONNX")
    ap.add_argument("--quick", action="store_true",
                    help="short run for smoke-testing the harness; NOT a reportable result")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    if args.quick:
        args.warmup, args.iters = 5, 30
        args.throughput_iters, args.onnx_eval_n, args.onnx_lat_iters = 3, 100, 10

    enable_windows_cuda_dlls()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True    # fixed shapes, as in training
        print(f"device: {torch.cuda.get_device_name(0)}")
        print("  NOTE: lock the clocks before a reportable run, e.g.\n"
              "        nvidia-smi -lgc <clock>   (consumer cards throttle under sustained load)")
    else:
        print("device: cpu  (latency numbers are not comparable to a GPU run)")

    pt_dir = PT_ROOT / args.task
    if not pt_dir.is_dir():
        print(f"no such folder: {pt_dir}")
        return 1
    models = sorted(p for p in pt_dir.glob("*.pt") if not p.name.endswith(".resume.pt"))
    if args.only:
        models = [p for p in models if p.stem in args.only]
    if not models:
        print("no checkpoints to benchmark")
        return 1
    print(f"models: {[m.stem for m in models]}\n")

    rows = []
    for i, p in enumerate(models, 1):
        print(f"  [{i}/{len(models)}] {p.stem}")
        try:
            rows += run_model(p, args.task, args, device)
        except Exception as e:
            print(f"      FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append(dict(model=p.stem, task=args.task, artifact="pytorch-fp32",
                             status="failed", error=f"{type(e).__name__}: {e}"[:300]))

    # ---- write ----
    out_dir = OUT_ROOT / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / ("baseline_gpu_quick.csv" if args.quick else "baseline_gpu.csv")
    try:
        import pandas as pd
        new = pd.DataFrame(rows)
        if csv.exists():
            old = pd.read_csv(csv)
            # Drop any existing row with the same (model, artifact) so a re-run REPLACES rather
            # than duplicates. Done with a set of tuples: a merge-with-indicator would add a
            # column and can collide with a real column name.
            if {"model", "artifact"}.issubset(old.columns):
                keys = set(zip(new["model"], new["artifact"]))
                keep = [(m, a) not in keys for m, a in zip(old["model"], old["artifact"])]
                old = old[keep]
            new = pd.concat([old, new], ignore_index=True)
        new.sort_values(["model", "artifact"]).to_csv(csv, index=False)
        print(f"\nwrote {csv}")

        if "status" in new.columns and "artifact" in new.columns:
            ok = new[(new.status == "ok") & (new.artifact == "pytorch-fp32")]
            cols = [c for c in ["model", "top1", "top5", "lat_p50_ms", "lat_p99_ms",
                                "throughput_img_s"] if c in ok.columns]
            if not ok.empty:
                print("\nFP32 baseline, ranked by top-1:")
                print(ok[cols].sort_values("top1", ascending=False).to_string(index=False))
                if "lat_p50_ms" in cols:
                    print("\nSame models, ranked by batch-1 latency:")
                    print(ok[cols].sort_values("lat_p50_ms").to_string(index=False))
    except ImportError:
        (out_dir / "baseline_gpu.json").write_text(json.dumps(rows, indent=2, default=str))
        print("  (pandas missing - wrote JSON instead)")

    for r in rows:
        if r.get("warning"):
            print(f"  WARNING  {r['model']}: {r['warning']}")
    bad = [r for r in rows if r.get("status") != "ok"]
    for r in bad:
        print(f"  FAILED   {r['model']}: {str(r.get('error', ''))[:160]}")
    if args.quick:
        print("\n--quick was used: these latencies are NOT reportable.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())