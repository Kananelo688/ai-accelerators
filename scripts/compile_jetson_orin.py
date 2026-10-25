#!/usr/bin/env python3
"""
compile_jetson_orin.py - ONNX -> TensorRT engine for the Jetson Orin Nano Super.

    checkpoint/onnx/<task>/<model>.onnx
        -> binaries/jetson/<task>/<model>_int8.engine   (or _fp16 / _fp32)
        -> binaries/jetson/<task>/<model>_int8.cache    (INT8 calibration cache, reusable)

*** RUN THIS ON THE JETSON ITSELF. ***
A TensorRT engine is not a portable file. It is specialised to the GPU architecture, the
TensorRT version, the CUDA/cuDNN versions and the driver on the machine that built it, and
TensorRT refuses to load an engine built elsewhere. There is no cross-compilation path for
Jetson in the normal workflow. So: clone the repo on the Orin, make sure checkpoint/onnx is
present, and run this there. The host details are written into the sidecar JSON precisely so a
mismatched engine can be identified later instead of producing a baffling load error.

The Orin Nano has NO DLA (the deep-learning accelerator is on the larger Orin modules), so
everything runs on the GPU through TensorRT. There is no DLA fallback flag here for that reason.

PRECISION
---------
  --precision int8   default. Uses the SHARED calibration set, so these numbers sit beside
                     Coral's and Hailo's on the same footing.
  --precision fp16   the precision most people actually deploy on Jetson, and a useful second
                     point: the INT8-vs-FP16 gap on the same hardware separates "quantisation
                     cost" from "hardware difference".
  --precision fp32   reference.

Note that TensorRT INT8 is not all-or-nothing: the builder keeps a layer in higher precision
whenever that is faster or when INT8 is unsupported for it. That is a genuine difference from
Coral and Hailo, which are INT8-only, and it must be stated when comparing them - a Jetson
"INT8" engine is really mixed precision chosen by the builder.

BEFORE BENCHMARKING, not part of this script: fix the power mode and clocks, or the numbers
are not reproducible.
    sudo nvpmodel -m 0        # MAXN SUPER (25W); use -m 1 for the 15W point
    sudo jetson_clocks

REQUIREMENTS: JetPack with TensorRT and PyCUDA (TensorRT ships with JetPack).
    pip install pycuda        # if not already present

Usage (on the Jetson)
---------------------
    python scripts/compile_jetson_orin.py --task classification --dry-run
    python scripts/compile_jetson_orin.py --task classification --precision int8
    python scripts/compile_jetson_orin.py --task classification --precision fp16

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec   # noqa: E402

BACKEND = "jetson"


def make_calibrator(trt, calib_u8: np.ndarray, cache_path: Path):
    """Build an IInt8EntropyCalibrator2 that feeds the shared calibration images.

    TensorRT drives calibration itself: it calls get_batch() repeatedly, runs the network, and
    records activation histograms. Our job is only to hand it correctly preprocessed data in
    GPU memory.

    EntropyCalibrator2 is TensorRT's recommended default for CNNs. It picks scales by minimising
    the information loss of the quantised distribution, rather than simply taking min/max. That
    is ALGORITHMICALLY different from what Coral and Hailo do, and the difference is deliberate:
    the common-recipe track fixes the calibration DATA and lets each vendor use its own default
    algorithm, because forcing a shared algorithm is not possible across these toolchains.
    """
    import pycuda.driver as cuda

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.data = ec.normalize_nchw(calib_u8)      # TensorRT is NCHW
            self.batch = 1                               # engines are built at static batch 1
            self.i = 0
            self.nbytes = self.data[0:1].nbytes
            # One reusable pinned-to-device buffer; allocating per batch would be wasteful and
            # can fragment the Orin's shared memory.
            self.d_input = cuda.mem_alloc(self.nbytes)
            self.cache_path = cache_path

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if self.i >= len(self.data):
                return None                              # None tells TensorRT calibration is done
            chunk = np.ascontiguousarray(self.data[self.i:self.i + self.batch])
            cuda.memcpy_htod(self.d_input, chunk)
            self.i += self.batch
            return [int(self.d_input)]

        def read_calibration_cache(self):
            # If a cache exists, TensorRT skips the calibration passes entirely. This makes a
            # rebuild (different precision, new TensorRT version) much faster AND guarantees the
            # identical scales are reused, which matters for reproducibility.
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return Calibrator()


def compile_one(onnx_path: Path, out_path: Path, args) -> dict:
    import tensorrt as trt

    res = ec.model_res(onnx_path, args.res)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH is required for ONNX-sourced networks; the implicit-batch path is legacy.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"[parse] TensorRT could not parse the ONNX graph: {errs[:400]}")

    config = builder.create_builder_config()
    # Workspace is scratch space for the tactic search, not the engine's runtime footprint. Too
    # small and TensorRT silently skips the fastest algorithms. The Orin Nano shares memory with
    # the CPU, so this is capped rather than generous.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mb * (1 << 20))

    row: dict = dict(res=res, precision=args.precision,
                     trt_version=trt.__version__, n_layers=network.num_layers)
    cache_path = out_path.with_suffix(".cache")

    if args.precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("this platform reports no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        # FP16 is enabled alongside INT8 on purpose: it gives the builder a sensible fallback
        # precision for layers it will not run in INT8, instead of dropping them to FP32.
        config.set_flag(trt.BuilderFlag.FP16)
        calib_u8 = ec.load_calibration(args.task_current, res, args.calib)[:args.calib_size]
        config.int8_calibrator = make_calibrator(trt, calib_u8, cache_path)
        row["calib_n"] = len(calib_u8)
    elif args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this platform reports no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("[build] TensorRT returned no engine; re-run with a verbose logger")
    out_path.write_bytes(serialized)

    # Deserialize once to read back what was actually built. Confirming the I/O shapes catches
    # a silently wrong resolution before it costs a whole benchmark run.
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    io = []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        io.append(dict(name=n, shape=list(engine.get_tensor_shape(n)),
                       dtype=str(engine.get_tensor_dtype(n)),
                       mode=str(engine.get_tensor_mode(n))))
    row["io"] = io

    ec.write_sidecar(out_path, row,
                     dict(chain=f"onnx -> TensorRT ({args.precision}) -> .engine",
                          calibration_cache=(str(cache_path.name) if cache_path.exists() else None),
                          portability_warning=(
                              "NOT PORTABLE. Valid only for this GPU architecture, TensorRT "
                              "version, CUDA version and driver. Rebuild on any other machine."),
                          int8_note=("TensorRT INT8 is mixed precision - the builder keeps some "
                                     "layers in FP16/FP32. Unlike Coral and Hailo, which are "
                                     "INT8-only. State this when comparing."
                                     if args.precision == "int8" else None)))
    return row


def main():
    ap = ec.base_parser(__doc__.split("\n")[1], BACKEND)
    ap.add_argument("--precision", choices=["int8", "fp16", "fp32"], default="int8")
    ap.add_argument("--workspace-mb", type=int, default=2048,
                    help="tactic-search scratch space, not the engine's runtime memory")
    args = ap.parse_args()

    # The artefact name encodes the precision, so int8 / fp16 / fp32 engines coexist and the
    # skip rule treats them independently.
    suffix = f"_{args.precision}.engine"

    ec.require(ec.have_module("tensorrt"),
               "tensorrt not importable. This script must run ON THE JETSON, where TensorRT "
               "ships with JetPack. Engines are not portable, so building them on the laptop "
               "would be pointless even if TensorRT were installed there.")
    if args.precision == "int8":
        ec.require(ec.have_module("pycuda"),
                   "pycuda not installed (needed to feed the INT8 calibrator).  pip install pycuda")

    # Importing pycuda.autoinit creates the CUDA context that the calibrator's mem_alloc needs.
    # Done here, once, rather than per model.
    if args.precision == "int8":
        import pycuda.autoinit  # noqa: F401

    import platform
    if platform.machine() != "aarch64":
        print(f"\n  WARNING: machine is {platform.machine()}, not aarch64. If this is not the "
              f"Jetson, the engines produced here will NOT load on it.\n")

    tasks = ec.TASKS if args.task == "all" else [args.task]
    rows = []
    for t in tasks:
        args.task_current = t
        rows += ec.compile_task(t, args, suffix, compile_one)
    return ec.finish(rows, args)


if __name__ == "__main__":
    sys.exit(main())