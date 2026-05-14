#!/bin/bash
# PAWS — build all CUDA/Cython extensions in the active env (PyTorch 2.x + Python 3.10).
# Tested with: Python 3.10, PyTorch 2.1.0+cu118, CUDA toolkit 11.7/11.8.
#
# NOTE on the install style:
# The original PLA repo recommends `python setup.py build develop` inside
# scene_graph_benchmark, which (a) compiles the maskrcnn_benchmark CUDA
# extension and (b) registers the package via pip's editable mode.
# In modern setuptools (>= 60), the `develop` command internally calls
# `pip install -e . --use-pep517`, which triggers an isolated build env
# that doesn't see the active torch — making the install fragile.
#
# We instead compile each extension in-place (`setup.py build_ext --inplace`)
# and let the user expose the packages via PYTHONPATH (see the message at
# the end of this script). Functionally identical to `build develop`.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "PAWS root: $ROOT"
echo

echo "[1/4] Building scene_graph_benchmark (maskrcnn_benchmark CUDA ops, AttrRCNN)..."
cd "$ROOT/third_party/scene_graph_benchmark"
python setup.py build_ext --inplace

echo
echo "[2/4] Building fasterRCNN/lib (RoI / NMS ops)..."
cd "$ROOT/third_party/fasterRCNN/lib"
python setup.py build_ext --inplace

echo
echo "[3/4] Building cython box_intersections (IoU helper)..."
cd "$ROOT/lib/fpn/box_intersections_cpu"
python setup.py build_ext --inplace

echo
echo "[4/4] Building cython draw_rectangles (spatial-mask helper)..."
cd "$ROOT/lib/draw_rectangles"
python setup.py build_ext --inplace

echo
echo "==============================================================="
echo "Build complete."
echo
echo "Add the following to your shell init (or run before training):"
echo
echo "  export PYTHONPATH=\$PYTHONPATH:$ROOT:$ROOT/third_party:$ROOT/third_party/scene_graph_benchmark"
echo
echo "This exposes:"
echo "  - lib.*, dataloader.*, scripts.*    (PAWS code)"
echo "  - fasterRCNN.lib.model.*             (third_party fasterRCNN)"
echo "  - scene_graph_benchmark.*, maskrcnn_benchmark.* (third_party scene_graph_benchmark)"
echo "==============================================================="
