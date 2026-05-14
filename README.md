# PAWS — Pair-Affinity Weakly-supervised Scene graph generation

PAWS extends the [PLA](https://github.com/zjucsq/PLA) weakly-supervised SGG framework with **three** complementary contributions:

- **RAM** (Reliability + Attention Matching): each VinVL detection is scored against Grounding-DINO's per-token attention to obtain a `reliability` score (cls-token) and a per-detection `match_score` (subject- / object-token). These scores then drive a bidirectional IoU propagation that converts a handful of annotated frames into pseudo-labels covering the entire video.
- **PA** (Pair Affinity learning): distance-aware BCE losses (`balanced_dist_bce`, `dist_bce`, `focal_dist`, `hnm_dist`) on top of the RAM pseudo-labels, teaching the model which subject–object pairs are likely to interact.
- **PAM** (Pair Affinity Masking): a learned attention bias inside the spatio-temporal transformer that suppresses uninformative pairs at inference.

Both **STTran** and **DSG-DETR** backbones are supported, and the entire pipeline runs in a single modern PyTorch environment.

## Highlights

- RAM-based weak supervision: VinVL + Grounding-DINO yield interaction-aware pseudo-labels without any scene-graph annotation at training time.
- Drop-in PA / PAM modules over an STTran / DSG-DETR scene graph generator.
- Patched maskrcnn-benchmark / fasterRCNN CUDA ops that **compile cleanly on PyTorch 2.x** (no separate legacy env required).

## Results (Action Genome, full test set, 1737 videos)

| Method | with R@10 / 20 / 50 | no R@10 / 20 / 50 |
|---|---|---|
| Baseline (no PA, no PAM) | 15.90 / 21.87 / 26.63 | 14.93 / 21.88 / 31.87 |
| **PAWS (PA + PAM)** | **22.24 / 26.48 / 28.00** | **23.21 / 30.24 / 37.47** |
| Δ over baseline | +6.34 / +4.61 / +1.37 | +8.28 / +8.36 / +5.60 |

Robustness to detector quality (2× / 4× downsampled VinVL features) is preserved — see `configs/eval_lowres/`.

## Repository layout

```
PAWS/
├── README.md, requirements.txt
├── configs/
│   ├── detector/        VinVL detector config
│   ├── pla_stage_1/     teacher (stage-1) configs
│   ├── pla_stage_2/     student (PA / PAM) configs
│   └── eval_lowres/     downsample-input eval configs
├── lib/
│   ├── sttran.py, dsg_detr.py               # backbones with PA / PAM heads
│   ├── transformer_wk.py, transformer_img.py # WS- and per-frame transformers
│   ├── transition_module.py                 # relation transition (PLA)
│   ├── assign_pseudo_label.py               # pseudo-label assignment + entry construction
│   ├── evaluation_recall.py                 # SGG R@K evaluator
│   ├── object_detector.py, extract_bbox_features.py
│   ├── loss.py                              # PA / PAM losses
│   ├── word_vectors.py                      # GloVe loader
│   ├── CSA/, draw_rectangles/, fpn/, ults/  # spatial masks, IoU, helpers
│   └── ...
├── dataloader/action_genome.py
├── scripts/
│   ├── install.sh                           # build all CUDA / Cython extensions
│   ├── train_sttran.py,  train_dsgdetr.py   # training entry-points (STTran / DSG-DETR)
│   ├── test_sttran.py,   test_dsgdetr.py    # evaluation entry-points
│   ├── test_sttran_full.py                  # full-set eval wrapper (R@K + mAP)
│   ├── convert_to_ag_class.py               # map VinVL / GDino classes → AG class space
│   └── generate_pseudo_labels.py            # RAM pipeline: produce gt_pla_gdino_*.pkl
├── third_party/
│   ├── scene_graph_benchmark/   maskrcnn_benchmark + AttrRCNN (PyTorch 2.x patched)
│   └── fasterRCNN/lib/          RoIAlign / RoIPool / NMS (PyTorch 2.x patched)
```

> All entry-point scripts (`train_*.py`, `test_*.py`) live under `scripts/` and are designed to be run **from the repo root** so they can find `configs/`, `lib/`, and `data/` via relative paths.

## Environment

Tested with:

| | version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0 + CUDA 11.8 |
| CUDA toolkit | 11.7 / 11.8 |
| GCC | 7+ |

```bash
conda create -n paws python=3.10 -y
conda activate paws

# PyTorch — match your local CUDA version
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Common deps (numpy<2 is pinned to match PyTorch 2.1's NumPy 1.x ABI)
pip install -r requirements.txt

# Build CUDA / Cython extensions in this env
bash scripts/install.sh    # 4 extensions: maskrcnn_benchmark, fasterRCNN, draw_rectangles, box_intersections

# Path
export PYTHONPATH=$(pwd):$(pwd)/third_party:$(pwd)/third_party/scene_graph_benchmark
```

## Data preparation

You need the Action Genome dataset, plus a precomputed VinVL detection cache and a Grounding-DINO–based pseudo-label file.

### 1. Action Genome frames + raw annotations

Follow the [Action Genome instructions](https://github.com/JingweiJ/ActionGenome) to obtain:

```
data/action-genome/
├── frames/             # extracted JPGs (~480p)
│   └── <video.mp4>/<frame_id>.png
├── annotations/
│   ├── person_bbox.pkl
│   ├── object_bbox_and_relationship.pkl
│   ├── object_classes.txt
│   ├── relationship_classes.txt
│   └── test_frames.json
```

### 2. VinVL detection cache (per-frame VinVL detections + 2048-d RoI features)

Either:

```bash
# Build cache from frames (requires VinVL weights; see configs/detector/)
python scripts/extract_vinvl_lowres_dets_feats.py \
    --frames_root data/action-genome/frames \
    --out_root data/action-genome/PLA_det_ag_class \
    --factor 1
```

or download the pre-extracted cache: [link TBD] (~50 GB for AG, native resolution).

For low-resolution evaluation, set `--factor 2` or `--factor 4`.

### 3. RAM pseudo-labels (Grounding-DINO derived)

The weakly-supervised pseudo-labels (`gt_pla_gdino_modelfree_05_skip2.pkl`) come from our **RAM** (Reliability + Attention Matching) pipeline:

```
annotated frames                       VinVL dets cache
        │                                       │
        ▼                                       │
┌──────────────────────────┐                    │
│ Grounding-DINO inference │                    │
│   per-token attention →  │                    │
│   reliability + match    │                    │
└──────────────────────────┘                    │
        │                                       │
        ▼                                       │
data/action-genome/PLA_gdino_12_5/   ←─── per-detection {rect, conf, class,
                                             reliability, match_score}
        │
        ▼  scripts/convert_to_ag_class.py
data/action-genome/PLA_gdino_12_5_ag_class/  (VinVL classes → AG classes,
                                              keeps reliability / match_score)
        │
        ▼  scripts/generate_pseudo_labels.py   (bidirectional IoU propagation,
                                                 RAM thresholds: reliability≥0.4,
                                                 match_score≥0.1)
data/action-genome/annotations/weak/gt_pla_gdino_modelfree_05_skip2.pkl
```

#### 3a. Grounding-DINO setup

```bash
# clone the official repo + checkpoint
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/GroundingDINO
cd third_party/GroundingDINO && pip install -e . --no-build-isolation && cd -

# download the official checkpoint (Objects365 + GoldG)
mkdir -p weights/groundingdino
cd weights/groundingdino
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd -
```

PAWS' `gdino` env already has `torch>=2.1` so GroundingDINO compiles in-place.

#### 3b. Run GDino matching → AG class space → pseudo-labels

```bash
# Step 1 — GDino matching on annotated AG frames (produces reliability + match_score per VinVL detection)
# (To be added — currently we ship the pre-built PLA_gdino_12_5_ag_class cache; see download link below.)

# Step 2 — VinVL/GDino classes → AG class space
python scripts/convert_to_ag_class.py \
    --src_det data/action-genome/PLA_gdino_12_5 \
    --src_feat data/action-genome/PLA_det_ag_class \
    --dst data/action-genome/PLA_gdino_12_5_ag_class

# Step 3 — bidirectional IoU propagation → final pseudo-label pickle
python scripts/generate_pseudo_labels.py \
    --config configs/pla_stage_1/sttran_ours.yml \
    --det_path data/action-genome/PLA_gdino_12_5_ag_class \
    --output data/action-genome/annotations/weak/gt_pla_gdino_modelfree_05_skip2.pkl
```

Pre-built artifacts (skip steps above): [link TBD] for `PLA_gdino_12_5_ag_class/` (~5 GB) and `gt_pla_gdino_modelfree_05_skip2.pkl` (~50 MB).

### 4. GloVe embeddings

Used for class word vectors.

```bash
mkdir -p lib/CSA/glove.6B
cd lib/CSA/glove.6B
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.200d.txt
```

The 200-dim version is sufficient; on first run the `.txt` is cached as `.pt`.

### 5. Pretrained model weights

| File | What it is | Where it goes |
|---|---|---|
| `vinvl_vg_x152c4.pth` | VinVL detector | `models/vinvl/` |
| `paws_stage1_sttran.tar` | Stage-1 teacher (per-frame) | `models/pla/` |
| `paws_stage1_dsgdetr.tar` | Stage-1 teacher (DSG-DETR variant) | `models/pla/` |
| `paws_stage2_sttran.tar` | Final PA+PAM model | `models/stage2/` |
| `paws_stage2_dsgdetr.tar` | Final PA+PAM model (DSG-DETR) | `models/stage2/` |

Pretrained weight downloads: [link TBD].

> For a quick path-agnostic setup, you can symlink an external data directory:
>
> ```bash
> ln -s /your/path/to/action-genome data/action-genome
> ln -s /your/path/to/PAWS-weights  models
> ```

## Training

### Stage 1 (teacher)

Trains a per-frame object-aware SGG model that emits soft predicate logits used to refine pseudo-labels.

```bash
python scripts/train_sttran.py  --cfg configs/pla_stage_1/sttran_ours.yml
python scripts/train_dsgdetr.py --cfg configs/pla_stage_1/dsgdetr_ours.yml
```

### Stage 2 (PAWS student, with PA + PAM)

Pair-affinity head and PAM-aware spatio-temporal transformer are turned on.

```bash
python scripts/train_sttran.py  --cfg configs/pla_stage_2/ours.yml
python scripts/train_dsgdetr.py --cfg configs/pla_stage_2/ours.yml
```

### Key Stage-2 knobs (`configs/pla_stage_2/ours.yml`)

| key | default | meaning |
|---|---|---|
| `pa_loss` | `['balanced_dist_bce']` | one of `dist_bce`, `balanced_dist_bce`, `focal_dist`, `hnm_dist`, `[]` (off) |
| `pam_loss` | `['adaptive']` | PAM training target: `adaptive`, `triplet`, `soft`, `[]` (off) |
| `pam` | `True` | apply PAM attention bias inside the transformer at inference |
| `pa_metric` | `True` | use the PA score to re-rank predicates at eval |
| `pa_weight` | `1.0` | weight for PA loss |
| `pam_weight` | `0.1` | weight for PAM loss |
| `pa_alpha_power` | `3.0` | distance-decay exponent for distance-weighted PA |
| `unmatched_sampling` | `True` | include unmatched proposals as negative pairs in PA training |
| `match` | `exact` | how detected boxes are matched to pseudo-labels |

## Evaluation

```bash
python scripts/test_sttran.py  --cfg configs/pla_stage_2/ours.yml          # STTran with PA / PAM
python scripts/test_dsgdetr.py --cfg configs/pla_stage_2/ours.yml          # DSG-DETR with PA / PAM
```

The evaluator reports R@10 / R@20 / R@50 / R@100 under with / semi / no constraint. If `pa_metric: True`, a second block also reports PA-gated re-ranked recalls.

For evaluation at lowered detector resolution, point to the matching detection cache and set `lowres_factor`:

```bash
python scripts/test_sttran.py --cfg configs/eval_lowres/lowres2x_ours_full.yml
python scripts/test_sttran.py --cfg configs/eval_lowres/lowres4x_ours_full.yml
```

## How the patches work (PyTorch 2.x)

The original `maskrcnn_benchmark` and `fasterRCNN/lib` CUDA extensions were authored against PyTorch 1.x and use APIs (`THC/THC.h`, `THCAtomics.cuh`, `THCudaCheck`, `THCCeilDiv`, `THCudaMalloc`) that were removed in PyTorch 1.11+. PAWS bundles patched copies under `third_party/`:

- `THC/THCAtomics.cuh` → `ATen/cuda/Atomic.cuh`
- `THC/THCDeviceUtils.cuh` → `ATen/cuda/DeviceUtils.cuh`
- `THCudaCheck(…)` → `C10_CUDA_CHECK(…)`
- `THCCeilDiv(a,b)` → inline `((a + b - 1) / b)`
- `THCudaMalloc / THCudaFree` → `at::empty(…)`-backed buffers (RAII)

These changes are confined to the CUDA / C++ entry-points; no Python or model logic was touched. The full patch is reproducible with the `sed` recipe in `scripts/install.sh`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `THC/THC.h: No such file` | Run `bash scripts/install.sh` again — patches must be re-applied if you replaced `third_party/`. |
| `ninja: command not found` | `pip install ninja` (faster build; not strictly required). |
| `CUDA mismatch: 11.7 vs 11.8` (warning) | Harmless if minor version differs. Hard failures require matching `cudatoolkit` and PyTorch CUDA build. |
| `cannot import name '_C' from 'maskrcnn_benchmark'` | The build copied `_C.*.so` to the wrong place. Re-run `python setup.py build_ext --inplace` inside `third_party/scene_graph_benchmark/`. |
| `module 'torch' has no attribute '_six'` | Already patched; if you see this, you may be running against an unpatched maskrcnn_benchmark on PYTHONPATH. Make sure `third_party/scene_graph_benchmark` precedes any old copies. |
| `np.float / np.int deprecated` | Already patched; check that `dataloader/`, `lib/`, and `third_party/` were copied from this repo (not from an upstream clone). |

## Acknowledgements

- **PLA** — pseudo-label assignment framework we extend.
- **STTran**, **DSG-DETR** — temporal-transformer backbones.
- **VinVL** — object detection / RoI feature extraction.
- **Grounding-DINO** — open-vocabulary object detection for pseudo-label sources.

## Citation

```bibtex
@inproceedings{paws,
  title={PAWS: Pair-Affinity Weakly-Supervised Scene Graph Generation},
  author={...},
  booktitle={...},
  year={2026}
}
```

## License

Apache License 2.0 (see `LICENSE`).
