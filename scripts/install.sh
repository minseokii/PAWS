#!/bin/bash
# PAWS: build all CUDA/Cython extensions in the active env (gdino-style: PyTorch 2.x + Python 3.10).
# Tested with: Python 3.10, PyTorch 2.1.0+cu118, CUDA toolkit 11.7/11.8.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "PAWS root: $ROOT"

echo "[1/4] Building scene_graph_benchmark (maskrcnn_benchmark + AttrRCNN ops)..."
cd "$ROOT/third_party/scene_graph_benchmark"
python setup.py build_ext --inplace

echo "[2/4] Building fasterRCNN/lib (RoI/NMS ops)..."
cd "$ROOT/third_party/fasterRCNN/lib"
python setup.py build_ext --inplace

echo "[3/4] Building cython box_intersections..."
cd "$ROOT/lib/fpn/box_intersections_cpu"
python setup.py build_ext --inplace

echo "[4/4] Building cython draw_rectangles..."
cd "$ROOT/lib/draw_rectangles"
python setup.py build_ext --inplace

echo "Done. Add PAWS root + third_party/scene_graph_benchmark to PYTHONPATH:"
echo "  export PYTHONPATH=\$PYTHONPATH:$ROOT:$ROOT/third_party/scene_graph_benchmark"
