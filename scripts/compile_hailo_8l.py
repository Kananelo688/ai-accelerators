#!/usr/bin/env python3
"""
compile_hailo_8l.py - ONNX -> HEF for the Hailo-8L, via the Hailo Dataflow Compiler.

    checkpoint/onnx/<task>/<model>.onnx
        -> parse     -> <model>.har              (Hailo Archive, float)
        -> optimize  -> <model>_quantized.har    (INT8, using the shared calibration set)
        -> compile   -> <model>.hef              (the deliverable)
        -> binaries/hailo-8l/<task>/

THE THREE STAGES, AND WHAT EACH CAN TELL YOU
--------------------------------------------
  parse     Translates the ONNX graph into Hailo's own representation. This is where an
            unsupported operator stops the build outright - the clearest possible
            deployability signal, and it costs seconds to discover.
  optimize  Post-training quantisation against the calibration set. Hailo reports a
            signal-to-noise ratio per layer; a low SNR means the architecture quantises badly
            even though it compiled, which is a different and subtler failure than a parse error.
  compile   Allocates the network across the Hailo-8L's compute resources. A model too large
            for one context is split into MULTIPLE CONTEXTS, which the device swaps at runtime -
            functional, but with a large latency penalty. The context count is recorded.

The API is used rather than the `hailo` CLI because the three stages share one ClientRunner
object, which keeps the calibration set in memory and gives structured access to the results.

hw_arch IS NOT OPTIONAL: the Hailo-8L is a reduced-capacity part, not a Hailo-8. Compiling for
"hailo8" produces a HEF that will not load on the 8L, so this script hard-codes "hailo8l".

REQUIREMENTS (x86_64 Linux only - the DFC has no ARM build, so this cannot run on the Pi that
hosts the accelerator; build here, copy the .hef over)
    * Hailo Dataflow Compiler 3.x  (the 3.x branch is the one that supports Hailo-8 / 8L;
      4.x and later target Hailo-10/15). Download from the Hailo Developer Zone - it is behind
      a free account, so it cannot be pip-installed from PyPI.
    * pip install onnxsim   (optional but recommended - see the simplify step below)

Usage
-----
    python scripts/compile_hailo_8l.py --task classification --dry-run
    python scripts/compile_hailo_8l.py --task classification
    python scripts/compile_hailo_8l.py --task classification --optimization-level 1

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec   # noqa: E402

BACKEND = "hailo-8l"
SUFFIX = ".hef"
HW_ARCH = "hailo8l"        # NOT "hailo8" - see the header


def simplify_onnx(onnx_path: Path, work: Path) -> Path:
    """Run onnxsim before parsing, if it is available.

    The Hailo parser is strict about graph shape. onnxsim folds constant subgraphs and collapses
    redundant Reshape/Transpose chains, which removes a common class of parse failure. It is a
    semantics-preserving rewrite, so it does not change what is being measured - but it IS a
    difference from the graph the other backends consume, so it is recorded in the sidecar.
    """
    if not ec.have_module("onnxsim"):
        return onnx_path
    try:
        import onnx
        from onnxsim import simplify
        model, ok = simplify(onnx.load(str(onnx_path)))
        if not ok:
            return onnx_path
        out = work / f"{onnx_path.stem}_simplified.onnx"
        onnx.save(model, str(out))
        return out
    except Exception as e:
        print(f"      onnxsim skipped ({type(e).__name__}); parsing the original graph")
        return onnx_path


def compile_one(onnx_path: Path, out_path: Path, args) -> dict:
    from hailo_sdk_client import ClientRunner

    out_dir = out_path.parent
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)

    res = ec.model_res(onnx_path, args.res)
    calib_u8 = ec.load_calibration(args.task_current, res, args.calib)[:args.calib_size]

    src = simplify_onnx(onnx_path, work) if not args.no_simplify else onnx_path
    row: dict = dict(res=res, precision="int8", calib_n=len(calib_u8),
                     hw_arch=HW_ARCH, simplified=(src != onnx_path))

    runner = ClientRunner(hw_arch=HW_ARCH)

    # ---- stage 1: parse ----
    # net_input_shapes pins the batch/shape explicitly. The exported graphs are already static at
    # batch 1, but stating it here means a shape mismatch fails with a clear message instead of
    # surfacing as a confusing allocation error during compile.
    try:
        runner.translate_onnx_model(
            str(src), onnx_path.stem,
            net_input_shapes={"input": [1, 3, res, res]},
        )
    except Exception as e:
        # Re-raised with the stage attached: knowing WHICH stage failed is the whole point of
        # separating them, since parse / optimize / compile failures mean very different things.
        raise RuntimeError(f"[parse] {type(e).__name__}: {e}") from e

    har_parsed = work / f"{onnx_path.stem}.har"
    runner.save_har(str(har_parsed))

    # ---- stage 2: optimize (quantise) ----
    # The DFC expects the calibration set as NHWC float. It applies its own normalisation only if
    # a normalization layer is declared in the model script; we are NOT declaring one, so the
    # images must arrive already normalised exactly as in training.
    calib = ec.normalize_nhwc(calib_u8)

    # A model script ("alls") tunes the optimisation. optimization_level is the main dial:
    #   0 = fastest, plain min/max       1 = default, with equalization
    #   2+ = slower, adds algorithms such as bias correction / fine-tuning
    # Level 1 is the default here because it is Hailo's own default - deviating from it per model
    # would be exactly the kind of per-model tuning the common-recipe track rules out.
    alls = [f"model_optimization_flavor(optimization_level={args.optimization_level})"]
    if args.alls:
        alls += Path(args.alls).read_text().splitlines()
    runner.load_model_script("\n".join(alls) + "\n")

    try:
        runner.optimize(calib)
    except Exception as e:
        raise RuntimeError(f"[optimize] {type(e).__name__}: {e}") from e

    har_quant = work / f"{onnx_path.stem}_quantized.har"
    runner.save_har(str(har_quant))
    row["har_mb"] = round(har_quant.stat().st_size / 2**20, 3)

    # ---- stage 3: compile ----
    try:
        hef = runner.compile()
    except Exception as e:
        raise RuntimeError(f"[compile] {type(e).__name__}: {e}") from e
    out_path.write_bytes(hef)

    # ---- post-compile introspection ----
    # hailortcli reports the context count, which is the number that explains an otherwise
    # inexplicable latency result: a multi-context model reloads weights between contexts.
    if ec.have_cmd("hailortcli"):
        code, info, _ = ec.run_cmd(["hailortcli", "parse-hef", str(out_path)],
                                   log_path=out_dir / f"{onnx_path.stem}.hef_info.log")
        if code == 0:
            n_ctx = info.lower().count("context")
            row["hef_contexts_mentioned"] = n_ctx
            if "multi" in info.lower() or n_ctx > 2:
                row["warning"] = ("HEF appears to use multiple contexts - expect a latency "
                                  "penalty from on-device weight swapping")
                print(f"      NOTE: {row['warning']}")

    if not args.keep_intermediate:
        har_parsed.unlink(missing_ok=True)
        if src != onnx_path:
            src.unlink(missing_ok=True)
        # The quantized HAR is kept regardless: it is what you re-open to inspect per-layer SNR
        # if a model's accuracy turns out to be poor.

    ec.write_sidecar(out_path, row,
                     dict(chain="onnx -> (onnxsim) -> HAR -> optimize(int8) -> HEF",
                          optimization_level=args.optimization_level,
                          quantized_har=str(har_quant.relative_to(ec.ROOT)),
                          calib_layout="NHWC float32, ImageNet-normalised",
                          note="hw_arch=hailo8l; a hailo8 HEF will not load on the 8L"))
    return row


def main():
    ap = ec.base_parser(__doc__.split("\n")[1], BACKEND)
    ap.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2, 3, 4],
                    help="Hailo optimisation effort; higher is slower to compile")
    ap.add_argument("--alls", type=Path, default=None,
                    help="extra Hailo model-script (.alls) lines to append")
    ap.add_argument("--no-simplify", action="store_true",
                    help="skip onnxsim and parse the original graph")
    args = ap.parse_args()

    ec.require(ec.have_module("hailo_sdk_client"),
               "Hailo Dataflow Compiler not installed. It is x86_64 Linux only and is downloaded "
               "from the Hailo Developer Zone (free account), not PyPI. Use the 3.x branch for "
               "Hailo-8L. This cannot run on the Raspberry Pi hosting the accelerator.")

    tasks = ec.TASKS if args.task == "all" else [args.task]
    rows = []
    for t in tasks:
        args.task_current = t
        rows += ec.compile_task(t, args, SUFFIX, compile_one)

    # Group failures by stage: a table of which architectures die at parse vs optimize vs compile
    # is a direct answer to "what is the compatibility cost of each lineage".
    bad = [r for r in rows if r["status"] != "ok"]
    if bad:
        print("\nFailure stage breakdown:")
        for stage in ("[parse]", "[optimize]", "[compile]"):
            hits = [r["model"] for r in bad if stage in r["error"]]
            if hits:
                print(f"  {stage:12s} {hits}")
        other = [r["model"] for r in bad if not any(s in r["error"] for s in
                                                    ("[parse]", "[optimize]", "[compile]"))]
        if other:
            print(f"  {'[other]':12s} {other}")
    return ec.finish(rows, args)


if __name__ == "__main__":
    sys.exit(main())