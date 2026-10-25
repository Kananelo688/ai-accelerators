#!/usr/bin/env python3
"""
compile_rpi_cpu.py - ONNX -> ONNX Runtime artefacts for the Raspberry Pi 5 CPU.

    checkpoint/onnx/<task>/<model>.onnx
        -> binaries/rpi-cpu/<task>/<model>_int8.onnx    (static INT8, QDQ)
        -> binaries/rpi-cpu/<task>/<model>_fp32.onnx    (reference copy)

WHAT "COMPILE" MEANS HERE
-------------------------
The Pi 5 has no accelerator, so there is no device-specific binary to produce. The CPU backend
is the control condition of the whole study: the one platform where every model is guaranteed
to run, against which accelerator speed-ups and compile failures are measured.

Two artefacts per model, on purpose:
  _fp32.onnx  the unquantised graph - the accuracy and latency reference.
  _int8.onnx  statically quantised with the SHARED calibration set, so the Pi's INT8 numbers
              sit on the same footing as Coral's and Hailo's.

Deliberately NOT done here:
  * Graph optimisation is left to ONNX Runtime at load time on the Pi. ORT's optimisations are
    target-specific (it fuses differently for ARM NEON than for x86 AVX), so optimising on an
    x86 laptop and shipping the result would bake in the wrong decisions. Shipping the plain
    graph and letting the Pi optimise it is both faster and more honest.
  * XNNPACK is a RUNTIME choice (an execution provider / delegate), not a build-time one. It is
    selected in the benchmark script, not here.

QUANTISATION FORMAT: QDQ (QuantizeLinear/DequantizeLinear nodes) rather than QOperator, because
QDQ is the portable representation - it keeps the graph readable, survives ORT version changes,
and is what the ARM CPU kernels expect.

Runs anywhere (x86 laptop or the Pi itself): the output is a portable ONNX file, not machine code.

Usage
-----
    python scripts/compile_rpi_cpu.py --task classification
    python scripts/compile_rpi_cpu.py --task classification --dry-run
    python scripts/compile_rpi_cpu.py --task classification --only mobilenetv2_100 --force

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec   # noqa: E402

BACKEND = "rpi-cpu"
SUFFIX = "_int8.onnx"      # the artefact whose presence marks a model as done


class NpyCalibrationReader:
    """Feeds the shared calibration images to onnxruntime.quantization.

    ORT asks for one dict of {input_name: array} per call and stops when we return None.
    Batch size 1 because the exported graphs have a static batch dimension of 1.
    """

    def __init__(self, data_u8: np.ndarray, input_name: str):
        # normalize_nchw applies the SAME ImageNet mean/std the model was trained with. Feeding
        # raw [0,255] pixels here would compute activation ranges for a distribution the model
        # never sees, and the resulting scales would be wrong.
        self.data = ec.normalize_nchw(data_u8)
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        x = self.data[self.i:self.i + 1]
        self.i += 1
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def compile_one(onnx_path: Path, out_path: Path, args) -> dict:
    """Quantise one model to static INT8 and copy the fp32 reference alongside it."""
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnxruntime as ort

    res = ec.model_res(onnx_path, args.res)
    calib = ec.load_calibration(args.task_current, res, args.calib)[:args.calib_size]

    # The fp32 reference is just a copy - no transformation, so it stays bit-identical to the
    # file the parity check in export_onnx.py validated.
    fp32_path = out_path.parent / f"{onnx_path.stem}_fp32.onnx"
    shutil.copy2(onnx_path, fp32_path)

    # Pre-processing runs shape inference and constant folding. Without it, quantize_static
    # frequently fails or silently leaves large parts of the graph in float, because it cannot
    # infer the shapes it needs to place QDQ nodes.
    prep_path = out_path.parent / f"{onnx_path.stem}.prep.onnx"
    quant_pre_process(str(onnx_path), str(prep_path), skip_symbolic_shape=False)

    input_name = ort.InferenceSession(
        str(prep_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    quantize_static(
        model_input=str(prep_path),
        model_output=str(out_path),
        calibration_data_reader=NpyCalibrationReader(calib, input_name),
        quant_format=QuantFormat.QDQ,
        # Activations unsigned, weights signed. This is the combination the ARM/XNNPACK INT8
        # kernels are written for; the signed/unsigned pair is not arbitrary.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # Per-channel weight scales. Depthwise convolutions have wildly different ranges per
        # channel, so a single per-tensor scale destroys accuracy on exactly the mobile models
        # this study is about. This is the most important flag in the file.
        per_channel=True,
        # MinMax is the most conservative and most comparable choice. Entropy and percentile
        # calibration are available, but each vendor's default differs anyway - the point of
        # the common-recipe track is to fix the DATA, and report the algorithmic differences.
        calibrate_method=CalibrationMethod.MinMax,
    )

    if not args.keep_intermediate:
        prep_path.unlink(missing_ok=True)

    # Count how much of the graph actually ended up quantised. A model where only a handful of
    # nodes are QDQ-wrapped is "INT8" in name only, and its latency will look suspiciously like
    # fp32 - worth catching here rather than puzzling over it in the results.
    import onnx
    g = onnx.load(str(out_path))
    ops: dict[str, int] = {}
    for n in g.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    n_q = ops.get("QuantizeLinear", 0) + ops.get("DequantizeLinear", 0)

    row = dict(res=res, precision="int8-qdq", calib_n=len(calib),
               n_nodes=len(g.graph.node), n_qdq_nodes=n_q,
               fp32_mb=round(fp32_path.stat().st_size / 2**20, 3),
               ort_version=ort.__version__)
    ec.write_sidecar(out_path, row, dict(op_types=ops, quant_format="QDQ", per_channel=True,
                                         calibrate_method="MinMax",
                                         note="Graph optimisation intentionally deferred to "
                                              "ONNX Runtime on the Pi (target-specific)."))
    return row


def main():
    ap = ec.base_parser(__doc__.split("\n")[1], BACKEND)
    args = ap.parse_args()

    # Fail before touching any model if the toolchain is missing.
    ec.require(ec.have_module("onnxruntime"),
               "onnxruntime not installed.  pip install onnxruntime")
    ec.require(ec.have_module("onnx"), "onnx not installed.  pip install onnx")

    tasks = ec.TASKS if args.task == "all" else [args.task]
    rows = []
    for t in tasks:
        args.task_current = t      # compile_one needs to know which calibration set to load
        rows += ec.compile_task(t, args, SUFFIX, compile_one)
    return ec.finish(rows, args)


if __name__ == "__main__":
    sys.exit(main())