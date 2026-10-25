#!/usr/bin/env python3
"""
prepare_classification_calibration.py - build the ONE INT8 calibration set that every backend will use.

    calib_datasets/<task>/calib_<res>.npy    uint8 NHWC, shape (N, res, res, 3)
    calib_datasets/<task>/calib_<res>.json   provenance: source, file list, seed, checksum

WHY IS THIS IMPORTANT
---------------------------
Coral, Hailo, TensorRT and ONNX Runtime each do their own post-training quantisation, and each
uses a different calibration ALGORITHM (per-tensor vs per-channel, min/max vs entropy vs
percentile). That difference is a finding worth reporting. What must NOT differ is the
calibration DATA - otherwise a cross-platform accuracy comparison is measuring the sample,
not the hardware. So the images are chosen once, here, and frozen on disk, every other model is quatized using this
clibration dataset.

The selection is a fixed-seed random sample, and the chosen files are recorded in the JSON,
so the exact set can be reconstructed and quoted in the writing.

This script is generic, and build calibration dataset for any task, irrespective of the backend or model.
The compile scripts for each backend will read the same calibration set, and use it to quantize their models.

SOURCE OF IMAGES
----------------
Always a directory of raw images, given with --images-dir. The directory may contain:

  * images directly                      -> a plain random sample is taken
  * sub-folders of images (class dirs)   -> a STRATIFIED sample is taken, spreading the quota
                                            as evenly as possible across sub-folders

Stratification matters on this dataset. The Kaggle ImageNet-100 train split is spread over
train.X1..X4, and each part holds a DIFFERENT 25 classes. A plain random sample over one part,
or an unstratified sample that happens to clump, would set the quantisation scales from a
fraction of the label space. Point --images-dir at the dataset root and every class contributes.

There is deliberately NO option to sample from the training notebook's uint8 cache. The cache is
stored at the TRAINING resolution (160px, centre-cropped once at build time), while the exported
ONNX models run at the EVALUATION resolution (224px). Calibrating at the wrong resolution, from
images that were already cropped by a different transform, produces activation ranges for a
distribution the model never sees. Decoding the original JPEGs here is the only way to guarantee
the calibration images go through exactly the same transform as inference.

IMPORTANT: calibration images must come from TRAINING data, never from the evaluation set that
accuracy is reported on. Calibrating on the test set leaks it into the model.

Usage
-----
    # whole dataset root: every class in every part contributes
    python scripts/prepare_classification_calibration.py --task classification --res 224 \
        --images-dir datasets/imageNet-100

    OR 
    
    # a single folder of images, larger sample
    python scripts/prepare_classification_calibration.py --task classification --res 224 \
        --images-dir datasets/imageNet-100/train.X1 --n 512


Author: Kananelo Chabeli
Affiliation: University of Cape Town
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_common import CALIB_ROOT, ROOT, TASKS   # noqa: E402

# 256 images is the usual sweet spot: Hailo's docs and the TFLite guides both suggest a few
# hundred. Below ~100 the activation ranges are noisy; above ~1000 the extra images change the
# computed scales very little while calibration time grows linearly.
DEFAULT_N = 256

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folders that are never calibration sources, however the root is pointed. Guards against
# accidentally calibrating on the evaluation split, which would leak it into the model.
EXCLUDE_DIRS = {"val", "val.x", "valid", "validation", "test"}


def collect_files(root: Path) -> dict[str, list[Path]]:
    """Walk the directory and group every image by the folder that directly contains it.

    The group key is the containing folder's path relative to the root, so on the Kaggle layout
    the groups come out as "train.X1/n01530575", "train.X2/n01692333" and so on - i.e. one group
    per class per part, which is the granularity we want to stratify over.

    Images sitting directly in the root are grouped under ".".
    """
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
            continue
        rel = p.parent.relative_to(root)
        parts = {q.lower() for q in rel.parts}
        if parts & EXCLUDE_DIRS:          # skip anything under a val/test folder
            continue
        groups[rel.as_posix()].append(p)

    if not groups:
        raise FileNotFoundError(f"no images found under {root}")
    # Sort within each group so the file order is identical on every OS and filesystem.
    return {k: sorted(v) for k, v in sorted(groups.items())}

def stratified_sample(groups: dict[str, list[Path]], n: int, seed: int) -> list[Path]:
    """Pick n files, spread as evenly as possible across the groups.

    With 100 classes and n=256 this gives every class 2 images and a randomly chosen 56 classes
    a third. The "which groups get the remainder" choice is itself seeded, so an alphabetically
    early class is no more likely to be over-represented than a late one.

    Groups smaller than their quota give everything they have, and the shortfall is redistributed
    over the remaining groups - so the total is always n unless the whole pool is smaller.
    """
    rng = np.random.default_rng(seed)
    keys = list(groups)
    total = sum(len(v) for v in groups.values())
    if n >= total:
        print(f"  requested {n} >= pool of {total}; using every image")
        return [f for k in keys for f in groups[k]]

    # Repeatedly hand out an equal quota, topping up from groups that still have images left.
    # A single pass is not enough when some groups are smaller than the quota.
    taken: dict[str, int] = {k: 0 for k in keys}
    remaining = n
    active = [k for k in keys if len(groups[k]) > 0]
    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        # The groups that get one extra image are chosen at random, not by position.
        bonus = set(rng.choice(len(active), size=extra, replace=False).tolist()) if extra else set()
        progressed = False
        for i, k in enumerate(list(active)):
            want = base + (1 if i in bonus else 0)
            can = min(want, len(groups[k]) - taken[k])
            if can > 0:
                taken[k] += can
                remaining -= can
                progressed = True
            if taken[k] >= len(groups[k]):
                active.remove(k)
        if not progressed:                 # every active group is exhausted
            break

    picked: list[Path] = []
    for k in keys:
        if taken[k]:
            # Choose WHICH images within the group randomly, then sort for deterministic order.
            idx = rng.choice(len(groups[k]), size=taken[k], replace=False)
            picked += [groups[k][i] for i in sorted(idx.tolist())]
    return sorted(picked)

def decode(files: list[Path], res: int) -> np.ndarray:
    """Decode the chosen images into one uint8 NHWC array, using the eval resize+crop.

    Mirrors the eval transform from the training notebook (resize short side to res/0.875, then
    centre-crop) so the calibration images are distributed like the images the model will see.
    This must match the inference preprocessing EXACTLY - a different interpolation or crop ratio
    shifts the activation histograms and the quantisation scales come out wrong in a way that is
    very hard to diagnose from accuracy numbers alone.
    """
    from PIL import Image
    import torchvision.transforms as T

    tf = T.Compose([T.Resize(int(round(res / 0.875))), T.CenterCrop(res)])
    out = np.empty((len(files), res, res, 3), np.uint8)
    for i, f in enumerate(files):
        # convert("RGB") handles the grayscale and CMYK files ImageNet contains, which would
        # otherwise produce 1- or 4-channel arrays and fail the assignment below.
        out[i] = np.asarray(tf(Image.open(f).convert("RGB")), np.uint8)
        if (i + 1) % 64 == 0 or i + 1 == len(files):
            print(f"  decoded {i + 1}/{len(files)}", end="\r")
    print()
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="dataset root: a folder of images, or a folder of sub-folders of images")
    ap.add_argument("--res", type=int, default=224, help="must match the ONNX input resolution")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="number of calibration images")
    ap.add_argument("--seed", type=int, default=1234, help="fixed, so the set is reproducible")
    ap.add_argument("--force", action="store_true", help="overwrite an existing calibration set")
    args = ap.parse_args()

    out_dir = CALIB_ROOT / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"calib_{args.res}.npy"

    if out_npy.exists() and not args.force:
        meta = json.loads(out_npy.with_suffix(".json").read_text())
        print(f"calibration set already exists: {out_npy}")
        print(f"  {meta['n']} images, {meta['res']}px, source={meta['source']}, seed={meta['seed']}")
        print("  use --force to regenerate (this INVALIDATES every INT8 binary already built)")
        return 0

    root = args.images_dir.resolve()
    print(f"scanning {root} ...")
    groups = collect_files(root)
    pool = sum(len(v) for v in groups.values())
    flat = len(groups) == 1 and next(iter(groups)) == "."
    print(f"  found {pool} images in {len(groups)} "
          f"{'folder (flat)' if flat else 'sub-folders -> stratified sampling'}")

    files = stratified_sample(groups, args.n, args.seed)
    print(f"  selected {len(files)} images "
          f"({len({f.parent for f in files})} distinct folders represented)")

    data = decode(files, args.res)
    np.save(out_npy, data)

    # A checksum of the array makes it trivial to prove later that two backends really did use
    # the same bytes this guarantees the common-recipe track for bechmaarking rests on.
    sha = hashlib.sha256(data.tobytes()).hexdigest()[:16]
    meta = dict(
        task=args.task, res=args.res, n=int(len(data)), seed=args.seed,
        dtype="uint8", layout="NHWC", range="[0,255] raw pixels, not normalised",
        sha256_16=sha,
        source=f"dir:{root.as_posix()}",
        pool_size=pool,
        n_groups=len(groups),
        sampling="flat-random" if flat else "stratified-by-subfolder",
        transform=f"Resize({int(round(args.res / 0.875))}) -> CenterCrop({args.res}) -> RGB uint8",
        # The exact file list IS the reproducibility claim. Relative paths so the record stays
        # valid when the dataset moves or another machine rebuilds it.
        files=[f.relative_to(root).as_posix() for f in files],
    )
    out_npy.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {out_npy}")
    print(f"  {len(data)} images, {args.res}px, uint8 NHWC, sha256[:16]={sha}")
    print(f"  {data.nbytes / 2**20:.1f} MB")
    print("\nThis is now the FROZEN calibration set for every backend on this task.")
    print("Regenerating it later invalidates all INT8 binaries built from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())