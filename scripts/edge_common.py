"""
edge_common.py - shared plumbing for the four compile_*.py scripts.

Everything in here is deliberately backend-agnostic: paths, task discovery, the
"only compile what is missing" rule, the calibration loader, and the per-backend CSV log.
Keeping it in one place is what makes the four backends comparable - they all read the SAME
ONNX files and the SAME calibration set, which is the premise of the common-recipe track.

Not a script. Imported by compile_coral_tpu.py, compile_hailo_8l.py,
compile_jetson_orin.py and compile_rpi_cpu.py.

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------------------
# Resolved from THIS FILE, not the cwd, so the scripts behave the same wherever they are run
# from - which matters because some of them run on the Jetson and some on the laptop.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

ONNX_ROOT = ROOT / "checkpoint" / "onnx"        # input: produced by export_onnx.py
BIN_ROOT = ROOT / "binaries"                    # output: one sub-tree per backend
CALIB_ROOT = ROOT / "calib-datasets"            # shared INT8 calibration sets (tracked in git)

TASKS = ["classification", "object-detection", "face-detection", "integrated", "object-tracking"]


# ---------------------------------------------------------------------------------------
# Calibration set
# ---------------------------------------------------------------------------------------
def load_calibration(task: str, res: int, path: Optional[Path] = None) -> np.ndarray:
    """Load the shared INT8 calibration images as a uint8 NHWC array.

    Produced by prepare_calibration.py. Using ONE file across all backends is the only way to
    claim that the INT8 comparison is like-for-like: the calibration ALGORITHM still differs per
    vendor (that difference is a result worth reporting), but the calibration DATA does not.

    Returns uint8 NHWC in [0,255]. Each backend applies its own normalisation and layout, because
    they disagree: TFLite wants NHWC, TensorRT and ONNX Runtime want NCHW.
    """
    p = path or (CALIB_ROOT / task / f"calib_{res}.npy")
    if not p.exists():
        raise FileNotFoundError(
            f"calibration set not found: {p}\n"
            f"  generate it first:\n"
            f"    python scripts/prepare_classification_calibration.py --task {task} "
            f"--res {res} --images-dir <dataset root>")
    arr = np.load(p)
    assert arr.ndim == 4 and arr.shape[3] == 3 and arr.dtype == np.uint8, \
        f"expected uint8 NHWC, got {arr.dtype} {arr.shape}"
    assert arr.shape[1] == arr.shape[2] == res, f"calib is {arr.shape[1]}px, asked for {res}px"
    return arr


# ImageNet-1k statistics, the same constants the training notebook used. Kept here rather than
# re-derived per backend, so a change applies everywhere at once.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def normalize_nchw(batch_u8: np.ndarray) -> np.ndarray:
    """uint8 NHWC [0,255] -> float32 NCHW, ImageNet-normalised. For TensorRT / ONNX Runtime."""
    x = batch_u8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def normalize_nhwc(batch_u8: np.ndarray) -> np.ndarray:
    """uint8 NHWC [0,255] -> float32 NHWC, ImageNet-normalised. For TFLite / Edge TPU."""
    x = batch_u8.astype(np.float32) / 255.0
    return np.ascontiguousarray((x - IMAGENET_MEAN) / IMAGENET_STD)


# ---------------------------------------------------------------------------------------
# ONNX sidecar metadata (written by export_onnx.py)
# ---------------------------------------------------------------------------------------
def read_sidecar(onnx_path: Path) -> dict:
    """Read the .json export_onnx.py wrote beside a model. Returns {} if it is missing.

    The sidecar carries the input resolution, the op histogram and the PyTorch accuracy, so each
    compile script can pick the right calibration resolution without being told.
    """
    p = onnx_path.with_suffix(".json")
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def model_res(onnx_path: Path, override: Optional[int], default: int = 224) -> int:
    """Resolution for this model: --res flag > the sidecar's recorded value > the default."""
    if override:
        return override
    return int(read_sidecar(onnx_path).get("res") or default)


# ---------------------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------------------
def have_cmd(cmd: str) -> bool:
    """True if an external executable is on PATH."""
    from shutil import which
    return which(cmd) is not None


def have_module(name: str) -> bool:
    """True if a Python module can be imported, without actually importing it."""
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def run_cmd(args: list[str], log_path: Optional[Path] = None, timeout: int = 3600):
    """Run an external tool, capture its output, and optionally tee it to a log file.

    The compiler LOGS are data, not noise: the Edge TPU compiler's log is where it says how many
    operations it mapped to the TPU and how many it pushed back to the CPU, which is the single
    most important deployability number in this study. So every log is kept on disk.
    """
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"$ {' '.join(args)}\n\n{out}")
    return proc.returncode, out, round(time.time() - t0, 1)


# ---------------------------------------------------------------------------------------
# Common CLI
# ---------------------------------------------------------------------------------------
def base_parser(description: str, backend: str) -> argparse.ArgumentParser:
    """The flags every compile script shares, so their interfaces stay identical."""
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS + ["all"],
                    help="which checkpoint/onnx/<task> folder to compile")
    ap.add_argument("--only", nargs="*", default=None, metavar="MODEL",
                    help="compile only these model names")
    ap.add_argument("--force", action="store_true",
                    help="recompile even if the output artefact already exists")
    ap.add_argument("--res", type=int, default=None,
                    help="override input resolution (default: the value in the ONNX sidecar)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="path to a calibration .npy (default: data/calibration/<task>/calib_<res>.npy)")
    ap.add_argument("--calib-size", type=int, default=256,
                    help="how many calibration images to actually use")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be compiled, check the toolchain, and stop")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="keep intermediate artefacts (saved_model, HAR, etc.) for debugging")
    ap.set_defaults(backend=backend)
    return ap


# ---------------------------------------------------------------------------------------
# Driver shared by all four backends
# ---------------------------------------------------------------------------------------
def compile_task(task: str, args, out_suffix: str,
                 compile_fn: Callable[[Path, Path, Any], dict]) -> list[dict]:
    """Walk one task folder and compile every pending model with the backend's compile_fn.

    out_suffix : the artefact extension that marks a model as done (e.g. "_edgetpu.tflite", ".hef")
    compile_fn : backend callback (onnx_path, out_path, args) -> dict of extra result fields.
                 It should RAISE on failure; this driver catches and records it.
    """
    in_dir = ONNX_ROOT / task
    out_dir = BIN_ROOT / args.backend / task

    if not in_dir.is_dir():
        print(f"[{task}] no ONNX folder: {in_dir} - skipping")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(in_dir.glob("*.onnx"))
    if args.only:
        models = [m for m in models if m.stem in args.only]

    # THE SKIP RULE, identical to export_onnx.py: an artefact on disk means done.
    pending = [m for m in models if args.force or not (out_dir / f"{m.stem}{out_suffix}").exists()]
    done = [m.stem for m in models if m not in pending]

    print(f"\n=== {args.backend} | task: {task} ===")
    print(f"  onnx models : {len(models)}")
    print(f"  already done: {len(done)} {done if done else ''}")
    print(f"  to compile  : {len(pending)} {[m.stem for m in pending]}")

    if args.dry_run:
        print("  (dry run - stopping here)")
        return []

    rows = []
    for i, m in enumerate(pending, 1):
        out_path = out_dir / f"{m.stem}{out_suffix}"
        print(f"\n  [{i}/{len(pending)}] {m.stem}")
        row: dict[str, Any] = dict(model=m.stem, task=task, backend=args.backend,
                                   status="failed", error="")
        t0 = time.time()
        try:
            row.update(compile_fn(m, out_path, args))
            row["status"] = "ok"
            if out_path.exists():
                row["size_mb"] = round(out_path.stat().st_size / 2**20, 3)
                # The backend wrote its sidecar before the artefact existed, so size_mb and
                # status were unknown then. Patch them in rather than make every backend
                # remember to do it.
                sc = out_path.with_suffix(".json")
                if sc.exists():
                    try:
                        meta = json.loads(sc.read_text())
                        meta.update(size_mb=row["size_mb"], status="ok")
                        sc.write_text(json.dumps(meta, indent=2, default=str))
                    except Exception:
                        pass
            print(f"      OK  -> {out_path.relative_to(ROOT)}"
                  + (f"  ({row['size_mb']} MB)" if "size_mb" in row else ""))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:400]
            traceback.print_exc()
            print(f"      FAILED: {row['error'][:200]}")
            # Delete a partial artefact so the skip rule does not treat it as a success.
            if out_path.exists():
                out_path.unlink()
        row["compile_s"] = round(time.time() - t0, 1)
        rows.append(row)
    return rows


def finish(rows: list[dict], args) -> int:
    """Print the summary, append the CSV log, and return a shell exit code."""
    print("\n" + "=" * 70)
    if not rows:
        print("nothing compiled (already done, or --dry-run). Use --force to redo.")
        return 0

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"[{args.backend}] compiled {len(ok)}/{len(rows)}   failed: {len(bad)}")
    for r in bad:
        print(f"  FAILED  {r['task']}/{r['model']}: {r['error'][:160]}")

    try:
        import pandas as pd
        for t in {r["task"] for r in rows}:
            csv = BIN_ROOT / args.backend / t / "compile_log.csv"
            new = pd.DataFrame([r for r in rows if r["task"] == t])
            if csv.exists():
                old = pd.read_csv(csv)
                old = old[~old.model.isin(new.model)]     # replace, do not duplicate
                new = pd.concat([old, new], ignore_index=True)
            new.sort_values("model").to_csv(csv, index=False)
            print(f"  log -> {csv.relative_to(ROOT)}")
    except ImportError:
        print("  (pandas not installed - skipped the CSV log)")

    return 1 if bad else 0


def host_info() -> dict:
    """Record WHERE a binary was built. TensorRT engines in particular are only valid on the
    machine/software stack that produced them, so this is not optional metadata."""
    return dict(host=platform.node(), machine=platform.machine(),
                system=f"{platform.system()} {platform.release()}",
                python=platform.python_version(),
                built_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def write_sidecar(out_path: Path, row: dict, extra: dict | None = None):
    """Write a .json beside the binary with everything needed to interpret it later."""
    meta = dict(row, **(extra or {}), **host_info())
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def require(ok: bool, msg: str):
    """Fail fast with an actionable message rather than a deep library traceback."""
    if not ok:
        print(f"\nTOOLCHAIN ERROR: {msg}\n", file=sys.stderr)
        sys.exit(2)