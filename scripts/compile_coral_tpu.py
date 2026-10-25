#!/usr/bin/env python3
"""
compile_coral_tpu.py - ONNX -> full-INT8 TFLite -> Edge TPU binary, for the Coral PCIe.

    checkpoint/onnx/<task>/<model>.onnx
        -> onnx2tf            -> <model>_saved_model/        (intermediate, deleted by default)
        -> TFLite converter   -> <model>_int8.tflite         (kept: the CPU-side reference)
        -> edgetpu_compiler   -> <model>_edgetpu.tflite      (the deliverable)
        -> binaries/tpu/<task>/

WHY THE CHAIN IS THIS LONG
--------------------------
The Edge TPU compiler accepts exactly one input format: a TFLite file quantised to FULL integer,
with INT8 weights AND INT8 activations. There is no ONNX path. So the model has to be carried
across frameworks, and each hop can change it.

The riskiest hop is ONNX -> TensorFlow. ONNX is NCHW; TensorFlow and TFLite are NHWC. onnx2tf
rewrites the layout, and where it cannot do so cleanly it inserts Transpose nodes. Those
transposes are a common reason an otherwise-supported model gets partially pushed back to the
CPU, so this script records how many survive.

THE COMPILER LOG IS THE RESULT
------------------------------
edgetpu_compiler prints how many operations it mapped to the Edge TPU and how many stayed on the
CPU. For this study that split is more valuable than the latency number it later produces: it is
the direct measure of "does this architecture actually fit a fixed-function accelerator". The log
is parsed into the CSV and kept verbatim on disk.

Two things to expect, and to report rather than work around:
  * The Edge TPU has roughly 8 MB of on-chip SRAM. A model whose parameters exceed it gets split,
    streaming weights off-host every inference, which costs far more than the model size suggests.
  * The compiler cuts the graph at the FIRST unsupported operation; everything after it runs on
    the CPU regardless of whether it was supported. So one bad op early is catastrophic, and the
    same op late is nearly free. The log gives the cut point.

REQUIREMENTS (x86_64 Linux only - edgetpu_compiler has no ARM build, so this cannot run on the Pi)
    pip install onnx2tf tensorflow onnx onnx_graphsurgeon sng4onnx onnxsim
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update && sudo apt install edgetpu-compiler

Usage
-----
    python scripts/compile_coral_tpu.py --task classification --dry-run
    python scripts/compile_coral_tpu.py --task classification
    python scripts/compile_coral_tpu.py --task classification --only squeezenet1_1 --force

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec   # noqa: E402

BACKEND = "tpu"
SUFFIX = "_edgetpu.tflite"


def parse_compiler_log(text: str) -> dict:
    """Pull the numbers that matter out of edgetpu_compiler's output.

    Written defensively with regexes because the exact wording has changed between compiler
    versions; a line that does not match simply leaves its field as None rather than crashing
    a 14-model run.
    """
    out: dict = {}

    def grab(pattern, cast=int):
        m = re.search(pattern, text, re.IGNORECASE)
        return cast(m.group(1).replace(",", "")) if m else None

    out["ops_total"] = grab(r"Number of operations that will run on (?:Edge TPU|CPU).*?:\s*(\d+)")
    out["ops_tpu"] = grab(r"operations that will run on Edge TPU:\s*([\d,]+)")
    out["ops_cpu"] = grab(r"operations that will run on CPU:\s*([\d,]+)")
    out["on_chip_mem_bytes"] = grab(r"On-chip memory used for caching model parameters:\s*([\d.]+)\s*(\w+)",
                                    cast=float)
    out["compiler_version"] = (m.group(1) if (m := re.search(r"Edge TPU Compiler version\s*([^\s]+)", text)) else None)

    if out["ops_tpu"] is not None and out["ops_cpu"] is not None:
        total = out["ops_tpu"] + out["ops_cpu"]
        out["ops_total"] = total
        # The headline deployability number: what fraction of the graph the accelerator accepted.
        out["tpu_frac"] = round(out["ops_tpu"] / total, 4) if total else 0.0

    # The compiler lists the first unsupported op and the reason it stopped there. That reason
    # is the single most useful line for the thesis, so capture it verbatim.
    reasons = re.findall(r"Operation is (?:working on an unsupported data type|otherwise supported, but not mapped due to some unspecified limitation|not supported)[^\n]*", text)
    out["unsupported_notes"] = "; ".join(dict.fromkeys(reasons))[:400] or None
    return out


def onnx_to_saved_model(onnx_path: Path, work_dir: Path) -> Path:
    """ONNX (NCHW) -> TensorFlow SavedModel (NHWC) via onnx2tf.

    onnx2tf is used rather than the older onnx-tf because it rewrites the layout properly instead
    of wrapping every convolution in Transpose pairs - which is the difference between a model
    the Edge TPU accepts and one it rejects.
    """
    sm_dir = work_dir / f"{onnx_path.stem}_saved_model"
    if sm_dir.exists():
        shutil.rmtree(sm_dir)
    code, out, _ = ec.run_cmd(
        ["onnx2tf", "-i", str(onnx_path), "-o", str(sm_dir),
         "-osd",            # output a SavedModel (not just .tflite files)
         "-coion",          # keep the original ONNX input/output names, so later tooling matches
         "-n"],             # non-verbose
        log_path=work_dir / f"{onnx_path.stem}.onnx2tf.log")
    if code != 0 or not sm_dir.exists():
        raise RuntimeError(f"onnx2tf failed (exit {code}); see the log. Tail: {out[-300:]}")
    return sm_dir


def saved_model_to_int8_tflite(sm_dir: Path, tflite_path: Path, calib_u8: np.ndarray) -> dict:
    """SavedModel -> TFLite, quantised to full integer with the shared calibration set."""
    import tensorflow as tf

    # NHWC here, not NCHW: TensorFlow's layout. Normalisation must match training, otherwise the
    # computed activation ranges describe a distribution the model never sees.
    calib = ec.normalize_nhwc(calib_u8)

    def representative_dataset():
        # One sample at a time - the converter expects a list of inputs per yield.
        for i in range(len(calib)):
            yield [calib[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm_dir))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    # FULL integer, enforced. Without this list the converter happily falls back to float for ops
    # it cannot quantise, producing a hybrid model that edgetpu_compiler will refuse.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # INT8 in and out. The Edge TPU can take float I/O and quantise on the host, but that adds a
    # conversion to every inference and would contaminate the latency measurement.
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    tflite_path.write_bytes(conv.convert())

    # Inspect what came out, so a silently-float-heavy model is caught now rather than later.
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    det = interp.get_tensor_details()
    n_int8 = sum(1 for d in det if d["dtype"] == np.int8)
    n_float = sum(1 for d in det if d["dtype"] in (np.float32, np.float16))
    return dict(tflite_tensors=len(det), tflite_int8_tensors=n_int8,
                tflite_float_tensors=n_float,
                tflite_in_dtype=str(interp.get_input_details()[0]["dtype"].__name__),
                tf_version=tf.__version__)


def compile_one(onnx_path: Path, out_path: Path, args) -> dict:
    out_dir = out_path.parent
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    res = ec.model_res(onnx_path, args.res)
    calib = ec.load_calibration(args.task_current, res, args.calib)[:args.calib_size]

    # --- hop 1: ONNX -> SavedModel ---
    sm_dir = onnx_to_saved_model(onnx_path, work)

    # --- hop 2: SavedModel -> full-INT8 TFLite (kept: this is the Coral CPU-fallback reference) ---
    tflite_path = out_dir / f"{onnx_path.stem}_int8.tflite"
    tf_info = saved_model_to_int8_tflite(sm_dir, tflite_path, calib)

    # --- hop 3: TFLite -> Edge TPU ---
    # -s prints the operation-mapping summary we want; -o sets the output directory. The compiler
    # names its output <stem>_edgetpu.tflite, which is exactly SUFFIX, so no rename is needed.
    log_path = out_dir / f"{onnx_path.stem}.edgetpu.log"
    code, log, _ = ec.run_cmd(
        ["edgetpu_compiler", "-s", "-o", str(out_dir), str(tflite_path)], log_path=log_path)
    stats = parse_compiler_log(log)
    if code != 0 or not out_path.exists():
        raise RuntimeError(
            f"edgetpu_compiler failed (exit {code}). {stats.get('unsupported_notes') or ''} "
            f"Tail: {log[-300:]}")

    if not args.keep_intermediate:
        shutil.rmtree(sm_dir, ignore_errors=True)

    row = dict(res=res, precision="int8-full", calib_n=len(calib), **tf_info, **stats,
               tflite_mb=round(tflite_path.stat().st_size / 2**20, 3))

    # A compiled-but-mostly-CPU model is not a failure - it is a finding - but it must be loud,
    # because its latency would otherwise be misread as "the Edge TPU is slow at this model".
    if stats.get("tpu_frac") is not None and stats["tpu_frac"] < 0.95:
        row["warning"] = (f"only {stats['tpu_frac']:.0%} of operations mapped to the Edge TPU; "
                          f"{stats.get('ops_cpu')} run on the host CPU")
        print(f"      WARNING: {row['warning']}")

    ec.write_sidecar(out_path, row,
                     dict(chain="onnx -> onnx2tf -> SavedModel -> TFLite(int8) -> edgetpu_compiler",
                          layout_note="onnx2tf converts NCHW to NHWC; input/output are int8",
                          compiler_log=str(log_path.relative_to(ec.ROOT))))
    return row


def main():
    ap = ec.base_parser(__doc__.split("\n")[1], BACKEND)
    args = ap.parse_args()

    ec.require(ec.have_cmd("edgetpu_compiler"),
               "edgetpu_compiler not on PATH. It is x86_64 Linux only - see the header of this "
               "file for the apt install lines. It cannot run on the Raspberry Pi.")
    ec.require(ec.have_cmd("onnx2tf"), "onnx2tf not on PATH.  pip install onnx2tf")
    ec.require(ec.have_module("tensorflow"), "tensorflow not installed.  pip install tensorflow")

    tasks = ec.TASKS if args.task == "all" else [args.task]
    rows = []
    for t in tasks:
        args.task_current = t
        rows += ec.compile_task(t, args, SUFFIX, compile_one)

    # Surface the mapping fractions together: this little table is the first real deployability
    # result of the project.
    ok = [r for r in rows if r["status"] == "ok" and r.get("tpu_frac") is not None]
    if ok:
        print("\nEdge TPU operation mapping:")
        for r in sorted(ok, key=lambda r: r["tpu_frac"]):
            print(f"  {r['model']:28s} {r['tpu_frac']:6.1%} on TPU "
                  f"({r.get('ops_tpu')}/{r.get('ops_total')} ops)")
    return ec.finish(rows, args)


if __name__ == "__main__":
    sys.exit(main())