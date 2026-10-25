#!/usr/bin/env python3
"""
prepare_classification_calibration.py - build the ONE INT8 calibration set that every backend will use.

    data/calibration/<task>/calib_<res>.npy    uint8 NHWC, shape (N, res, res, 3)
    data/calibration/<task>/calib_<res>.json   provenance: source, indices, seed, checksum

WHY IS THIS IMPORTANT
---------------------------
Coral, Hailo, TensorRT and ONNX Runtime each do their own post-training quantisation, and each
uses a different calibration ALGORITHM (per-tensor vs per-channel, min/max vs entropy vs
percentile). That difference is a finding worth reporting. What must NOT differ is the
calibration DATA - otherwise a cross-platform accuracy comparison is measuring the sample,
not the hardware. So the images are chosen once, here, and frozen on disk, every other model is quatized using this
clibration dataset.

The selection is a fixed-seed random sample, and the chosen indices are recorded in the JSON,
so the exact set can be reconstructed and quoted in the writing.

This script is generic, and build calibration dataset for any task, irrespective of the backend or model.
The compile scripts for each backend will read the same calibration set, and use it to quantize their models.

SOURCE OF IMAGES
----------------
  --source cache  (default) the uint8 memmap the training notebook built, which guarantees the
                  calibration images went through EXACTLY the same resize/crop as validation.
  --source dir    a folder of raw images (for tasks with no cache, e.g. detection later).

IMPORTANT: calibration images must come from TRAINING data, never from the evaluation set that
accuracy is reported on. Calibrating on the test set leaks it into the model.

Usage
-----
    python scripts/prepare_classification_calibration.py --task classification --res 224
    python scripts/prepare_classification_calibration.py --task classification --res 224 --n 512
    python scripts/prepare_classification_calibration.py --task object-detection --source dir \
        --images-dir /path/to/images --res 320

            
Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_common import CALIB_ROOT, ROOT, TASKS   # noqa: E402

# Where the training notebook put its uint8 caches.
CACHE_DIR = ROOT / "datasets" / "imageNet-100_cache"

# 256 images is the usual sweet spot: Hailo's docs and the TFLite guides both suggest a few
# hundred. Below ~100 the activation ranges are noisy; above ~1000 the extra images change the
# computed scales very little while calibration time grows linearly.
DEFAULT_N = 256


def from_cache(split: str, res: int) -> np.ndarray:
    """Load the training notebook's uint8 NHWC memmap for one split."""
    p = CACHE_DIR / f"{split}_{res}_x.npy"
    if not p.exists():
        avail = sorted(q.name for q in CACHE_DIR.glob("*_x.npy")) if CACHE_DIR.is_dir() else []
        raise FileNotFoundError(
            f"no cache at {p}\n  available: {avail}\n"
            f"  The cache is built by the training notebook. If you are on a machine without it,\n"
            f"  use --source dir --images-dir <folder of images> instead.")
    return np.load(p, mmap_mode="r")      # memmap: we only read N rows out of it


def from_dir(images_dir: Path, res: int, n: int, seed: int) -> np.ndarray:
    """
        Decode a folder of images into uint8 NHWC, using the standard eval resize+crop.

        Mirrors the eval transform from the training notebook (resize short side to res/0.875, then
        centre-crop) so the calibration images are distributed like the images the model will see.

    """
    from PIL import Image
    import torchvision.transforms as T

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"no images under {images_dir}")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
    tf = T.Compose([T.Resize(int(round(res / 0.875))), T.CenterCrop(res)])

    out = np.empty((len(pick), res, res, 3), np.uint8)
    for i, idx in enumerate(sorted(pick)):
        out[i] = np.asarray(tf(Image.open(files[idx]).convert("RGB")), np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--res", type=int, default=224, help="must match the ONNX input resolution")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="number of calibration images")
    ap.add_argument("--seed", type=int, default=1234, help="fixed, so the set is reproducible")
    ap.add_argument("--source", choices=["cache", "dir"], default="cache")
    ap.add_argument("--split", default="train",
                    help="cache split to sample from - keep this on train, never val")
    ap.add_argument("--images-dir", type=Path, default=None, help="required when --source dir")
    ap.add_argument("--force", action="store_true", help="overwrite an existing calibration set")
    args = ap.parse_args()

    out_dir = CALIB_ROOT / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"calib_{args.res}.npy"

    if out_npy.exists() and not args.force:
        meta = json.loads(out_npy.with_suffix(".json").read_text())
        print(f"calibration set already exists: {out_npy.relative_to(ROOT)}")
        print(f"  {meta['n']} images, {meta['res']}px, source={meta['source']}, seed={meta['seed']}")
        print("  use --force to regenerate (this INVALIDATES every INT8 binary already built)")
        return 0

    if args.source == "cache":
        if args.split.lower() != "train":
            print(f"  WARNING: sampling calibration images from '{args.split}'. Calibrating on "
                  f"the evaluation split leaks it into the quantised model.")
        pool = from_cache(args.split, args.res)
        rng = np.random.default_rng(args.seed)
        # Sorted indices: a memmap is far faster to read in order than at random.
        idx = np.sort(rng.choice(len(pool), size=min(args.n, len(pool)), replace=False))
        data = np.ascontiguousarray(pool[idx])
        provenance = dict(source=f"cache:{args.split}_{args.res}", pool_size=int(len(pool)),
                          indices=idx.tolist())
    else:
        if not args.images_dir:
            ap.error("--source dir requires --images-dir")
        data = from_dir(args.images_dir, args.res, args.n, args.seed)
        provenance = dict(source=f"dir:{args.images_dir}", pool_size=None, indices=None)

    np.save(out_npy, data)

    # A checksum of the array makes it trivial to prove later that two backends really did use
    # the same bytes this guarantees the common-recipe track for bechmaarking rests on.
    sha = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    meta = dict(task=args.task, res=args.res, n=int(len(data)), seed=args.seed,
                dtype="uint8", layout="NHWC", range="[0,255] raw pixels, not normalised",
                sha256_16=sha, **provenance)
    out_npy.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {out_npy.relative_to(ROOT)}")
    print(f"  {len(data)} images, {args.res}px, uint8 NHWC, sha256[:16]={sha}")
    print(f"  {data.nbytes / 2**20:.1f} MB")
    print("\nThis is now the FROZEN calibration set for every backend on this task.")
    print("Regenerating it later invalidates all INT8 binaries built from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())