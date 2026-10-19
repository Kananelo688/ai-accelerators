"""
quantize_yolo_edgetpu.py

RUN ON:   x86-64 Ubuntu laptop (NOT the Raspberry Pi)
PRODUCES: yolov8n_full_integer_quant_edgetpu.tflite  -> scp this to the Pi

Pipeline:
  STAGE 1: PyTorch (.pt)        -> ONNX (.onnx)            [boxes normalized to 0-1]
  STAGE 2: ONNX                 -> TensorFlow SavedModel   [NCHW -> NHWC]
  STAGE 3: SavedModel           -> INT8 TFLite             [calibration with real images]
  STAGE 4: INT8 TFLite          -> Edge TPU TFLite         [edgetpu_compiler, x86-64 only]
"""
import os
import sys
import glob
import shutil
import importlib
import subprocess
import numpy as np
import cv2
import torch
import onnx
import onnxslim
import onnxruntime as ort
import tensorflow as tf
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.export.tensorflow import tf_wrapper
import struct
import onnx.helper

IMGSZ = 640  # preferrable to be multiples of 32, because The TPU uses 32 x 32 Systolic array of 8-bit processing elements.
CALIB_DIR = "/home/chabeli/Private/ai-accelerators/datasets/calib_dataset" #This is PATH where CALIBRATION IMAGES ARE PLACED.
ONNX_FILE = "yolov8n.onnx"
SAVED_MODEL_DIR = "yolov8n_saved_model"
OUTPUT_TFLITE = "yolov8n_INT8.tflite"
QUANTIZATION_PARAMS_FILE = "yolov8n_quant_params.txt"  # contains input/output scale and zero-point for the Pi

# NEW: one preprocessing function used everywhere (calibration, sanity check).
# IMPORTANT: your inference code on the Pi must preprocess images the SAME way,
# otherwise the int8 scale factors learned during calibration won't match.
def load_image(path):
    img = cv2.imread(path)                       # OpenCV loads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # model was trained on RGB
    img = cv2.resize(img, (IMGSZ, IMGSZ))
    return img.astype(np.float32) / 255.0        # HWC, 0-1 (same scaling as training)



# NEW: fail fast if there are no calibration images, before doing any work
calib_paths = sorted(
    glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
    + glob.glob(os.path.join(CALIB_DIR, "*.png"))
)
if not calib_paths:
    sys.exit(f"❌ No calibration images found in '{CALIB_DIR}/'")
print(f"Found {len(calib_paths)} calibration images in '{CALIB_DIR}/'")


# ==============================================================================
# STAGE 1: EXPORT PYTORCH GRAPH TO INTERMEDIATE ONNX FORMAT
# ==============================================================================
print("\n--- STAGE 1: Exporting PyTorch (.pt) -> ONNX (.onnx) ---")

# 1. Load trained FP32 PyTorch weights
pt_model = YOLO("yolov8n.pt")

#print class of the the model:
print(f"Model Class: {type(pt_model)}")

# Note: YOLO is a high-level wrapper. The actual torch.nn.Module (DetectionModel)
# lives in pt_model.model. We export that directly so we control the Detect head.
# fuse() merges each Conv + BatchNorm pair into a single Conv (fewer ops, same math).
torch_model = pt_model.model.fuse().eval()

# NEW: WHY CHANGE THE DETECT HEAD?
# YOLOv8's output tensor (1, 84, 8400) concatenates two very different things:
#   rows 0-3  : box x, y, w, h in PIXELS  (range ~0 to 640)
#   rows 4-83 : class scores              (range 0 to 1)
# INT8 gives ONE scale factor (S) per tensor. For a range of 0-640, one int8 step
# is ~2.5, so every class score (0-1) collapses to 0 -> almost no detections.
# tf_wrapper() swaps the head's box-decoding function for one that divides boxes
# by the image size, so the whole tensor is in the 0-1 range and int8 keeps
# precision for both. This is the same helper Ultralytics' own TFLite export uses.
# On the Pi you multiply boxes by 640 again to get pixels.
torch_model = tf_wrapper(torch_model)

for m in torch_model.modules():
    if isinstance(m, Detect):
        m.export = True      # return only the final (1, 84, 8400) tensor
        m.format = "tflite"  # tell the head we're exporting for TFLite
        m.dynamic = False    # Edge TPU requires static input shapes, dynamic=False, enforces this.


# 2. Export to ONNX.
# WHY dynamic=False? Edge TPU requires static input shapes (e.g. 1x3x640x640)
# to fix memory layout and allocate static SRAM buffers at compilation time.
# NEW: with torch.onnx.export the shape is fixed by the dummy input below.
# No dynamic_axes are given, so every dimension stays static.
dummy_input = torch.zeros(1, 3, IMGSZ, IMGSZ)  # NCHW, the layout PyTorch uses

with torch.no_grad():
    torch_model(dummy_input)  # dry run so the head builds its anchor grid for 640x640

torch.onnx.export(
    torch_model,
    dummy_input,
    ONNX_FILE,
    opset_version=13,          # use older opset to avoid unsupported-op errors in onnx2tf
    input_names=["images"],
    output_names=["output0"],
    do_constant_folding=True,  # pre-computes parts of the graph that never change
    dynamo=False,              # classic exporter (newer torch versions default to dynamo)
)

# Fuses redundant math nodes to clean up the graph
# NEW: this is what simplify=True did inside Ultralytics' export (onnxslim)
onnx.save(onnxslim.slim(onnx.load(ONNX_FILE)), ONNX_FILE)

# NEW: SANITY CHECK - run the ONNX model once and confirm boxes are now 0-1
sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
sample = np.ascontiguousarray(load_image(calib_paths[0]).transpose(2, 0, 1)[None])  # HWC -> NCHW + batch
out = sess.run(None, {"images": sample})[0]                                         # (1, 84, 8400)
print(f"ONNX output shape: {out.shape}")
print(f"Box value max:   {out[0, :4].max():.3f}  (should be about 1.0 or less)")
print(f"Score value max: {out[0, 4:].max():.3f}  (between 0 and 1)")
if out[0, :4].max() > 2.0:
    sys.exit("❌ Boxes are still in pixels: the Detect head change did not apply")

print(f"✅ Stage 1 Complete: Saved static ONNX model to {ONNX_FILE}")


# ==============================================================================
# STAGE 2: CONVERT ONNX GRAPH TO TENSORFLOW SAVEDMODEL
# ==============================================================================
print("\n--- STAGE 2: Converting ONNX -> TensorFlow SavedModel ---")

# NEW: onnx2tf always tries to download a small sample image file
# (20 images, 128x128x3, float32, 0-1) to sanity-check the converted graph.
# On the Pi that download returned something that isn't a valid .npy file,
# which caused "Cannot load file containing pickled data". We replace the
# download function with one that returns the same shape from our own images.
def local_test_image_data():
    imgs = [cv2.resize(load_image(p), (128, 128)) for p in calib_paths[:20]]
    return np.stack(imgs).astype(np.float32)   # shape [N, 128, 128, 3]

# We use onnx2tf to translate ONNX operators into native
# TensorFlow 2.x Keras layers/ops while preserving NCHW -> NHWC channel order.

# onnx_graphsurgeon (used inside onnx2tf) still calls onnx.helper.float32_to_bfloat16,
# which was removed in ONNX 1.17. Ultralytics adds it back the same way before calling onnx2tf.

if not hasattr(onnx.helper, "float32_to_bfloat16"):
    def float32_to_bfloat16(fval):
        ival = struct.unpack("=I", struct.pack("=f", fval))[0]  # float32 bits as an integer
        return ival >> 16                                        # keep the top 16 bits = bfloat16
    onnx.helper.float32_to_bfloat16 = float32_to_bfloat16


# called through its Python API instead of os.system(), so we can swap out
# the download function before conversion starts. It is patched in both modules,
# because onnx2tf.py imports its own copy of the function.
o2t_module = importlib.import_module("onnx2tf.onnx2tf")
cf_module = importlib.import_module("onnx2tf.utils.common_functions")
o2t_module.download_test_image_data = local_test_image_data
cf_module.download_test_image_data = local_test_image_data

# any error inside convert() raises an exception and stops the script,
# so a failed conversion can no longer print "Stage 2 Complete".
o2t_module.convert(
    input_onnx_file_path=ONNX_FILE,
    output_folder_path=SAVED_MODEL_DIR,
    output_signaturedefs=True,   # NEW: adds a serving signature to the SavedModel
    not_use_onnxsim=True,        # NEW: skip onnxsim, Stage 1 already simplified the graph
)

# NEW: double-check the SavedModel really exists before moving on
if not os.path.exists(os.path.join(SAVED_MODEL_DIR, "saved_model.pb")):
    sys.exit("❌ Stage 2 failed: saved_model.pb was not created")
print(f"✅ Stage 2 Complete: Created TensorFlow SavedModel at '{SAVED_MODEL_DIR}'")


# ==============================================================================
# STAGE 3: FULL INT8 POST-TRAINING QUANTIZATION (PTQ) VIA TFLITE CONVERTER
# ==============================================================================
print("\n--- STAGE 3: Running INT8 Calibration & Quantization ---")

# 1. Initialize the TFLite Converter from the SavedModel directory
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)


# 2. Define the Representative Dataset Generator Function
# WHY IS THIS NEEDED?
# Neural network activations (the outputs of ReLU/Conv layers) are continuous floats.
# To map float values (e.g., -2.41 to +5.83) into an 8-bit integer range (-128 to 127),
# the converter needs to "see" real sample data to calculate scale factors (S)
# and zero-point offsets (Z) for every internal layer tensor.
def representative_data_gen():
    # Pass up to 300 sample inputs through the network graph to calibrate dynamic range
    # NEW: real images instead of np.random.rand(). Random noise gives wrong
    # activation ranges, so S and Z would not match what the drone camera sees.
    for p in calib_paths[:300]:
        # Shape must match model expected shape [1, 640, 640, 3] (NHWC,
        # because onnx2tf converted the graph from NCHW to NHWC).
        data = np.expand_dims(load_image(p), 0)
        yield [data]

# 3. Enable Default TFLite Graph Optimizations (Constant folding, layout optimization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Attach the Calibration Dataset Generator
converter.representative_dataset = representative_data_gen

# 5. Enforce STRICT INT8 Execution for Edge TPU Compatibility
# Edge TPU hardware cannot process FP32 math. If any layer falls back to FP32,
# it will execute on the Pi's ARM CPU instead of the Coral accelerator.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# 6. Force Input and Output Tensors to INT8
# By default, TFLite keeps input/output tensors as FP32 (quantizing only hidden layers).
# Edge TPU requires hardware-level INT8 input/output buffers.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# 7. Run the actual conversion pass
print("Calibrating layer ranges and converting weights/activations to INT8...")
tflite_quant_model = converter.convert()


# 8. Write out the final quantized binary TFLite file
with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_quant_model)

# NEW: print the input/output scale (S) and zero-point (Z).
# On the Pi you need them to:
#   input : int8_value = float_value / S + Z     (image 0-1 -> int8)
#   output: float_value = (int8_value - Z) * S   (int8 -> boxes/scores)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter
interpreter = Interpreter(model_path=OUTPUT_TFLITE)
with open(QUANTIZATION_PARAMS_FILE, "w") as f:
    for d in interpreter.get_input_details() + interpreter.get_output_details():
        scale, zero_point = d["quantization"]
        f.write(f"{d['name']}: S={scale:.6f}, Z={zero_point}\n")


print(f"✅ Stage 3 Complete: Full INT8 Quantized model saved to '{OUTPUT_TFLITE}'")


# ==============================================================================
# STAGE 4: COMPILE THE INT8 TFLITE MODEL FOR THE EDGE TPU
# ==============================================================================
print("\n--- STAGE 4: Compiling INT8 TFLite -> Edge TPU TFLite ---")

# NEW: WHY IS THIS NEEDED?
# The INT8 .tflite from Stage 3 still runs on a normal CPU. edgetpu_compiler
# finds every op the TPU supports and replaces that part of the graph with one
# "edgetpu-custom-op" that runs on the Coral. Ops it can't map stay on the Pi's CPU.
# The compiler only exists for x86-64 Linux, which is why this script runs on the laptop.
if shutil.which("edgetpu_compiler") is None:
    sys.exit("❌ edgetpu_compiler not found. Install it with: sudo apt-get install edgetpu-compiler")

# -s : print a per-operation summary (mapped to TPU vs left on CPU)
subprocess.run(
    ["edgetpu_compiler", "-s", "--out_dir", ".", OUTPUT_TFLITE],
    check=True,
)

edgetpu_file = OUTPUT_TFLITE.replace(".tflite", "_edgetpu.tflite")
if not os.path.exists(edgetpu_file):
    sys.exit(f"❌ Stage 4 failed: '{edgetpu_file}' was not created")

print(f"✅ Stage 4 Complete: Edge TPU model saved to '{edgetpu_file}'")
