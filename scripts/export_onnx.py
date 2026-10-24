#!/usr/bin/env python3
"""
export_onnx.py - Stage 1 of the edge deployment pipeline: yTorch checkpoiPnt -> ONNX.

The filestructure the script expects is:
    ai-accelerators/
      checkpoint/
        pytorch/<task>/<model>.pt     <-- INPUT   (written by the training notebooks)
        onnx/<task>/<model>.onnx      <-- OUTPUT  (+ <model>.json sidecar)
      scripts/
        export_onnx.py                <-- this file

ONNX is the hub format of the benchmark: every downstream toolchain (TensorRT for Jetson,
ONNX Runtime / TFLite for the Raspberry Pi, the Edge TPU compiler for Coral, the Hailo
Dataflow Compiler) consumes ONNX or something derived from it. This script DELIBERATELY
stops at ONNX so that each stage can be validated in isolation.

The script is generic and converts any PyTorch checkpoint to ONNX. Target checkpoints must be specified
as a task corresponding to sub-folder in /checkpoint/pythorch.
The script is safe to re-run: it skips any checkpoint that already has a matching .onnx file, 
so it can be used to resume a partially-complete export. Use --force to re-export everything.

Usage
-----
    python scripts/export_onnx.py --task classification
    python scripts/export_onnx.py --task classification --force            # re-export everything
    python scripts/export_onnx.py --task classification --only mobilenetv2_100 ghostnet_100
    python scripts/export_onnx.py --task classification --opset 13 --res 224
    python scripts/export_onnx.py --task all                               # every task folder

By default only checkpoints WITHOUT a matching .onnx file are exported, so the script is
safe to re-run and cheap to resume.

Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------
# Resolve ROOT from THIS FILE rather than from the current working directory, so the script
# behaves identically whether it is run as `python scripts/export_onnx.py` from the project
# root or as `python export_onnx.py` from inside scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PT_ROOT = ROOT / "checkpoint" / "pytorch"
ONNX_ROOT = ROOT / "checkpoint" / "onnx"

# The task folders. A task name is just a sub-folder name shared by the input and output trees,
# which is what keeps this script generic: adding a new task needs no code change, only a folder.
TASKS = ["classification", "object-detection", "face-detection", "integrated", "object-tracking"]

# ---------------------------------------------------------------------------------------
# Export defaults
# ---------------------------------------------------------------------------------------

# The OPSET (Operator Set) in ONNX ( Open  Neural Network Exchange) is version-controlled
# Catelog of standard machine learning operation and thier mathematical specifications. 
# Each opset version introduces new operators and may deprecate or modify existing ones.

# OPSET 13 is the default because it is the highest version that every toolchain in this study
# accepts comfortably. Newer opsets are not automatically better here: the Hailo Dataflow
# Compiler and the ONNX->TFLite converters lag the ONNX spec, and an opset they do not know
# forces them to fall back or refuse. Raise it only if a specific model needs a newer op.


DEFAULT_OPSET = 13


# Per-model minimum opset. A model lands here when it uses an op that the default opset predates,
# and the entry documents WHY - the requirement is a finding about the model, not a config detail.
#
# mobilenetv4_hybrid_medium: its MultiQueryAttention2d blocks call
#   aten::scaled_dot_product_attention, which the ONNX exporter only supports from opset 14.
#   At opset 14 SDPA DECOMPOSES into MatMul / Softmax / Transpose / Reshape rather than becoming a
#   fused Attention node. That decomposition is what gives it any chance on the Edge TPU and Hailo,
#   since neither supports a fused attention op - but the 48 Transpose and 32 Reshape nodes it adds
#   are exactly the data-movement pattern those compilers struggle with. Worth watching at the
#   compile stage: this is the deployability test case.
OPSET_OVERRIDES = {
    "mobilenetv4_hybrid_medium": 14,
}

# STATIC batch size 1 is the default, NOT a dynamic axis. Fixed-function accelerators (Edge TPU,
# Hailo-8L) compile a static dataflow graph and cannot accept a symbolic batch dimension; TensorRT
# can, but then needs optimisation profiles. Batch 1 is also the latency case the UAV cares about.
# Use --dynamic-batch only to produce a throughput-sweep variant for the GPU/CPU backends.
DEFAULT_BATCH = 1

# Fallback input resolution, used only when the checkpoint does not record one.
DEFAULT_RES = 224 #This iis the standard Image Resolution for most models in the Literature.


# ---------------------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------------------
def load_checkpoint(path: Path) -> dict:
    """Load a .pt file onto the CPU and return it as a dict.

    weights_only=False is required because our checkpoints carry metadata (the config dict,
    the class list) alongside the tensors. PyTorch >= 2.6 defaults this to True, which would
    refuse to unpickle anything but tensors.

    SECURITY NOTE: weights_only=False executes pickle, so only ever point this script at
    checkpoints you produced yourself. That is the case here (as models are trained from scratch), 
    but it matters if external-repo weights are ever added.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # A checkpoint may also be a bare state_dict (common for third-party repos), in which case
    # wrap it so the rest of the code has one shape to deal with.
    if not isinstance(obj, dict) or "state_dict" in obj and "model" not in obj:
        obj = {"model": obj.get("state_dict", obj)} if isinstance(obj, dict) else {"model": obj}
    return obj


def extract_state_dict(ck: dict) -> dict:
    """Pull the weight tensors out of a checkpoint, under any of the common key names."""
    for key in ("model", "state_dict", "model_state_dict", "net"):
        sd = ck.get(key)
        # A state_dict is a mapping of name -> tensor. Guard on the value type so we do not
        # mistake a metadata dict (e.g. the config) for weights.
        if isinstance(sd, dict) and sd and all(torch.is_tensor(v) for v in sd.values()):
            return sd
    # Last resort: the checkpoint IS the state_dict.
    if ck and all(torch.is_tensor(v) for v in ck.values()):
        return ck
    raise KeyError(f"no state_dict found; top-level keys = {list(ck)[:10]}")


def clean_state_dict(sd: dict) -> dict:
    """Strip wrapper prefixes that would otherwise make load_state_dict fail.

    - "_orig_mod."  : added by torch.compile
    - "module."     : added by DataParallel / DistributedDataParallel
    Our training notebook saves the uncompiled module, so this is defensive - but it costs
    nothing and makes the script usable on checkpoints from other sources.
    """
    for prefix in ("_orig_mod.", "module."):
        if all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def get_config(ck: dict) -> dict:
    """Return the training config as a plain dict, whatever form it was saved in."""
    cfg = ck.get("config", {})
    if is_dataclass(cfg):
        cfg = asdict(cfg)
    return cfg if isinstance(cfg, dict) else {}


def infer_num_classes(sd: dict) -> Optional[int]:
    """Guess the number of output classes from the shape of the final weight tensor.

    Needed for checkpoints that carry no metadata (third-party weights, bare state_dicts).
    The classifier is the LAST parameter in a state_dict for every architecture here, and its
    leading dimension is the class count - for both Linear (out,in) and the 1x1 Conv head that
    SqueezeNet uses (out,in,1,1). We read the last *weight* rather than the last tensor,
    because biases and BN buffers can trail it.
    """
    for key in reversed(list(sd)):
        t = sd[key]
        if torch.is_tensor(t) and t.ndim >= 2 and key.endswith("weight"):
            return int(t.shape[0])
    return None


# ---------------------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------------------
# A state_dict is only tensors - it carries no architecture. To rebuild the module we need the
# architecture name and how it was constructed, which our training notebook stored in the
# checkpoint (name / source / kwargs / num_classes). That is exactly why those fields were saved.
def build_model(name: str, source: Optional[str], num_classes: int, kwargs: dict):
    """Instantiate the architecture. Mirrors build_model() in the training notebook.

    Imports are done lazily so that a task whose models come from neither library (a future
    external-repo detector, say) does not pay for importing timm.
    """
    if source in (None, "", "timm"):
        import timm
        if timm.is_model(name):
            return timm.create_model(name, pretrained=False, num_classes=num_classes, **kwargs)
    if source in (None, "", "torchvision"):
        import torchvision
        if name in torchvision.models.list_models(module=torchvision.models):
            return torchvision.models.get_model(name, weights=None, num_classes=num_classes, **kwargs)
    raise ValueError(f"cannot rebuild architecture '{name}' (source={source!r})")


def rebuild_from_checkpoint(ck: dict, fallback_name: str):
    """Rebuild the trained module from a checkpoint. Returns (model_in_eval_mode, metadata)."""
    cfg = get_config(ck)
    # Fall back to the filename if the checkpoint has no "name" field - true for third-party
    # weights, where the filename is usually the only clue to the architecture.
    name = ck.get("name") or fallback_name
    source = ck.get("source")
    kwargs = ck.get("kwargs") or {}
    if isinstance(kwargs, str):           # the summary CSV stores kwargs as a JSON string
        kwargs = json.loads(kwargs or "{}")

    sd = clean_state_dict(extract_state_dict(ck))
    # Prefer the recorded class count; fall back to reading it off the weights. Getting this
    # wrong builds the right architecture with the wrong head, which load_state_dict then
    # rejects - so inferring it is what makes metadata-free checkpoints work.
    num_classes = ck.get("num_classes") or cfg.get("num_classes") or infer_num_classes(sd) or 1000

    model = build_model(name, source, num_classes, kwargs)

    # strict=True on purpose: a silent key mismatch would export an architecture carrying
    # partly-random weights, which looks fine in ONNX and produces meaningless accuracy later.
    model.load_state_dict(sd, strict=True)

    # eval() matters for export, not just for inference:
    #   - BatchNorm switches to its running statistics, so the folded constants are the trained
    #     ones. Exporting in train mode would bake in batch statistics from the dummy input.
    #   - Dropout becomes the identity and disappears from the graph.
    model.eval()

    meta = dict(name=name, source=source, num_classes=num_classes, kwargs=kwargs,
                recipe_hash=ck.get("recipe_hash"), top1=ck.get("top1"), top5=ck.get("top5"),
                train_res=cfg.get("train_res"), eval_res=cfg.get("eval_res"))
    return model, meta


def resolve_resolution(meta: dict, override: Optional[int]) -> int:
    """Decide the ONNX input resolution.

    Priority: --res flag > the checkpoint's eval_res > the default.
    We use EVAL resolution, not train resolution: 160 was a training-budget choice, while the
    deployed model is evaluated (and benchmarked) at 224. The resolution is baked into the graph
    because the input shape is static, so this choice is not reversible after export - which is
    why it is recorded in the sidecar JSON.
    """
    if override:
        return override
    return int(meta.get("eval_res") or DEFAULT_RES)


# ---------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------
def export_onnx(model, out_path: Path, res: int, batch: int, opset: int,
                dynamic_batch: bool) -> None:
    """Trace the model and write an ONNX graph.

    torch.onnx.export runs the model once on a dummy input and records the operations.
    Because this is a TRACE, any Python control flow that depends on the input values is
    frozen as it happened on this one input. That is fine for these CNNs, which are static
    feed-forward graphs, but it is a real limitation to remember for detectors with NMS
    or trackers with data-dependent loops in later stages of this pipeline.
    """
    dummy = torch.randn(batch, 3, res, res)

    # Dynamic axes name the dimensions that may vary at runtime. We normally pass none, giving a
    # fully static graph - required by the Edge TPU compiler and preferred by Hailo.
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

    # PyTorch >= 2.9 warns that the TorchScript exporter is legacy. We are choosing it on
    # purpose (see dynamo=False below), so the warning is noise that would hide real problems.
    warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],         # fixed names make every downstream script uniform
        output_names=["output"],
        opset_version=opset,
        # Constant folding evaluates subgraphs that depend only on weights. Its most important
        # effect here is folding BatchNorm into the preceding convolution, which is what the
        # accelerator compilers expect to see and what the published MAC counts assume.
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        # Use the legacy TorchScript exporter rather than the newer dynamo path: it is the one
        # the edge toolchains have been tested against, and it produces flatter graphs.
        dynamo=False,
    )


def check_graph(path: Path) -> dict:
    """Validate the ONNX file and report a few facts worth recording.

    onnx.checker verifies the graph is well-formed (shapes, types, topological order). It does
    NOT tell us whether the maths matches PyTorch - that is what verify_parity does.
    """
    import onnx
    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    # Count how often each operator type appears. This is genuinely useful later: the op mix is
    # what decides whether a model compiles cleanly for the Edge TPU or Hailo, so capturing it
    # here saves re-parsing every graph during the deployability analysis.
    ops: dict[str, int] = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return dict(
        ir_version=m.ir_version,
        opset=max((imp.version for imp in m.opset_import), default=None),
        n_nodes=len(m.graph.node),
        n_op_types=len(ops),
        op_types=dict(sorted(ops.items(), key=lambda kv: -kv[1])),
    )


def verify_parity(model, path: Path, res: int, batch: int, n_trials: int = 3) -> dict:
    """Run PyTorch and ONNX Runtime on identical inputs and compare the outputs.

    This is the step that catches a BROKEN export - an op that traced incorrectly, a frozen
    branch, a wrong input layout. Without it a corrupted graph would sail through to the
    hardware benchmark and show up only as mysteriously poor accuracy, where it would be very
    hard to attribute.

    Reported:
      max_abs_diff   - largest elementwise difference in the logits
      top1_agree     - fraction of inputs where both pick the same class
      cosine         - cosine similarity of the logit vectors
    """
    import onnxruntime as ort

    # CPUExecutionProvider explicitly: we want to validate the GRAPH, not a particular backend,
    # and the CPU provider is the reference implementation.
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    max_diff, agree, cos_sims = 0.0, 0, []
    for _ in range(n_trials):
        # Random input, not zeros: a graph can be wrong in ways that produce identical output
        # on an all-zero input (e.g. a dead branch) while diverging on real data.
        x = torch.randn(batch, 3, res, res)
        with torch.no_grad():
            ref = model(x).cpu().numpy()
        got = sess.run(None, {in_name: x.numpy()})[0]

        max_diff = max(max_diff, float(np.abs(ref - got).max()))
        agree += int((ref.argmax(1) == got.argmax(1)).all())
        a, b = ref.ravel(), got.ravel()
        cos_sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))

    return dict(max_abs_diff=max_diff,
                top1_agree=agree / n_trials,
                cosine=float(np.mean(cos_sims)))


# ---------------------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------------------
def export_one(pt_path: Path, out_dir: Path, args) -> dict:
    """Export a single checkpoint. Returns a result row (never raises)."""
    stem = pt_path.stem                                   # "mobilenetv2_100.pt" -> "mobilenetv2_100"
    onnx_path = out_dir / f"{stem}.onnx"
    row: dict[str, Any] = dict(model=stem, task=out_dir.name, status="failed", error="")
    t0 = time.time()
    

    # An explicit --opset on the command line always wins; otherwise raise the default to the
    # model's minimum if it has one. Never lower it: the default is already the most
    # conservative value the edge toolchains accept.
    opset = args.opset
    if args.opset == DEFAULT_OPSET:
        opset = max(opset, OPSET_OVERRIDES.get(stem, 0))
        if opset != DEFAULT_OPSET:
            print(f"      opset raised to {opset} (required by this architecture)")
    try:
        ck = load_checkpoint(pt_path)
        model, meta = rebuild_from_checkpoint(ck, fallback_name=stem)
        res = resolve_resolution(meta, args.res)

        export_onnx(model, onnx_path, res, args.batch, opset, args.dynamic_batch)

        graph = check_graph(onnx_path)
        parity = verify_parity(model, onnx_path, res, args.batch) if not args.no_verify else {}

        row.update(
            status="ok",
            source=meta.get("source"),
            res=res,
            batch=args.batch,
            opset=graph["opset"],
            n_nodes=graph["n_nodes"],
            n_op_types=graph["n_op_types"],
            size_mb=round(onnx_path.stat().st_size / 2**20, 2),
            torch_top1=meta.get("top1"),
            recipe_hash=meta.get("recipe_hash"),
            **{f"parity_{k}": v for k, v in parity.items()},
        )

        # A warning rather than a failure: tiny numerical differences are normal (different
        # kernels, different accumulation order), but disagreement on the ARGMAX is not.
        if parity and parity["top1_agree"] < 1.0:
            row["error"] = "WARNING: top-1 disagreement between PyTorch and ONNX Runtime"

        # Sidecar JSON: everything needed to interpret this .onnx without opening it or
        # consulting the notebook. The op histogram feeds the later deployability analysis.
        sidecar = dict(row, exported_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       torch_version=torch.__version__,
                       source_checkpoint=str(pt_path.relative_to(ROOT)),
                       input_shape=[args.batch, 3, res, res],
                       dynamic_batch=args.dynamic_batch,
                       num_classes=meta.get("num_classes"),
                       op_types=graph["op_types"])
        (out_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))

    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        traceback.print_exc()
        # Remove a partial .onnx, otherwise the skip logic on the next run would treat this
        # broken file as a completed export.
        if onnx_path.exists():
            onnx_path.unlink()

    row["export_s"] = round(time.time() - t0, 1)
    return row


def run_task(task: str, args) -> list[dict]:
    """Export every pending checkpoint for one task."""
    pt_dir = PT_ROOT / task
    out_dir = ONNX_ROOT / task

    if not pt_dir.is_dir():
        print(f"[{task}] no such folder: {pt_dir} - skipping")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(p for p in pt_dir.glob("*.pt") if not p.name.endswith(".resume.pt"))
    if args.only:
        ckpts = [p for p in ckpts if p.stem in args.only]

    # THE SKIP RULE: a checkpoint is pending unless a matching .onnx already exists.
    pending = [p for p in ckpts if args.force or not (out_dir / f"{p.stem}.onnx").exists()]
    done = [p.stem for p in ckpts if p not in pending]

    print(f"\n=== task: {task} ===")
    print(f"  checkpoints : {len(ckpts)}")
    print(f"  already onnx: {len(done)} {done if done else ''}")
    print(f"  to export   : {len(pending)} {[p.stem for p in pending]}")

    rows = []
    for i, p in enumerate(pending, 1):
        print(f"\n  [{i}/{len(pending)}] {p.stem}")
        row = export_one(p, out_dir, args)
        rows.append(row)
        if row["status"] == "ok":
            par = (f" | parity dmax={row.get('parity_max_abs_diff', 0):.2e} "
                   f"top1={row.get('parity_top1_agree', float('nan')):.0%}"
                   if not args.no_verify else "")
            print(f"      OK  {row['size_mb']} MB, {row['n_nodes']} nodes, "
                  f"{row['res']}px, opset {row['opset']}{par}")
            if row["error"]:
                print(f"      {row['error']}")
        else:
            print(f"      FAILED: {row['error']}")
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Export trained PyTorch checkpoints to ONNX, one task folder at a time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS + ["all"],
                    help="which checkpoint/pytorch/<task> folder to process")
    ap.add_argument("--only", nargs="*", default=None, metavar="MODEL",
                    help="export only these model names (checkpoint stems)")
    ap.add_argument("--force", action="store_true",
                    help="re-export even if the .onnx already exists")
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                    help="ONNX opset version")
    ap.add_argument("--res", type=int, default=None,
                    help="override input resolution (default: the checkpoint's eval_res)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="static batch size baked into the graph")
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="make the batch dimension dynamic (NOT supported by Edge TPU / Hailo)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the PyTorch vs ONNX Runtime parity check (not recommended)")
    args = ap.parse_args()

    # Single thread: exports run sequentially and we want the timings to be comparable, not to
    # fight each other for the laptop's cores.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(0)       # the parity check uses random inputs; fix them for reproducibility

    tasks = TASKS if args.task == "all" else [args.task]
    rows: list[dict] = []
    for t in tasks:
        rows += run_task(t, args)

    # ---- summary ----
    print("\n" + "=" * 70)
    if not rows:
        print("nothing to do - every checkpoint already has an .onnx (use --force to redo)")
        return 0

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    warn = [r for r in ok if r["error"]]
    print(f"exported {len(ok)}/{len(rows)}   failed: {len(bad)}   warnings: {len(warn)}")
    for r in bad:
        print(f"  FAILED  {r['task']}/{r['model']}: {r['error']}")
    for r in warn:
        print(f"  WARN    {r['task']}/{r['model']}: {r['error']}")

    # One CSV per run, appended to the task folder, so the export history lives beside the
    # artefacts it describes. pandas is imported lazily to keep the script usable without it.
    try:
        import pandas as pd
        for t in {r["task"] for r in rows}:
            csv = ONNX_ROOT / t / "onnx_export.csv"
            new = pd.DataFrame([r for r in rows if r["task"] == t])
            if csv.exists():
                old = pd.read_csv(csv)
                # Replace rows for models exported again, rather than accumulating duplicates.
                old = old[~old.model.isin(new.model)]
                new = pd.concat([old, new], ignore_index=True)
            new.sort_values("model").to_csv(csv, index=False)
            print(f"  log -> {csv.relative_to(ROOT)}")
    except ImportError:
        print("  (pandas not installed - skipped the CSV log)")

    # Non-zero exit code so this can be chained in a shell script or CI without silently passing.
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())