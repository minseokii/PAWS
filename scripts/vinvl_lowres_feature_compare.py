"""
Feature-level smoke test: How do VinVL RoI features change under low-res input?

For each frame, extract bbox+feat at native and at 4× downsample.
For each native detection, find best-IoU matched low-res detection (same class).
Compute:
  - bbox IoU
  - score delta
  - 2048-dim RoI feature cosine similarity (the actual signal consumed by our SGG head)

Reports per-resolution avg cosine similarity, IoU, and score change.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PLA_ROOT = '/SSD1/minseok/WS-DSGG/TRKT/PLA'
sys.path.insert(0, os.path.join(PLA_ROOT, 'lib'))
sys.path.insert(0, os.path.join(PLA_ROOT, 'scene_graph_benchmark'))

from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from scene_graph_benchmark.config import sg_cfg


def load_model():
    config_file = os.path.join(PLA_ROOT, 'configs/detector/vinvl_x152c4.yaml')
    weight_file = os.path.join(PLA_ROOT, 'models/vinvl/vinvl_vg_x152c4.pth')
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHT = weight_file
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()
    DetectronCheckpointer(cfg, model, save_dir='./').load(cfg.MODEL.WEIGHT)
    transforms = build_transforms(cfg, is_train=False)
    return model, transforms


def detect_with_features(model, transforms, cv2_img):
    """Run VinVL forward and extract per-detection 2048-dim avg-pooled RoI features
    in the ORIGINAL image's coordinate frame. Features come from the final detected boxes.
    """
    H, W = cv2_img.shape[:2]
    img_input = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img_input, _ = transforms(img_input, target=None)
    in_h, in_w = img_input.shape[1], img_input.shape[2]

    with torch.no_grad():
        # Run full detector to get final boxes
        prediction = model([img_input.to(model.device)])[0].to('cpu')
        # boxes are in input-resolution; resize to original
        pred_orig = prediction.resize((W, H))
        boxes_orig = pred_orig.bbox.numpy()  # (N, 4)
        classes = pred_orig.get_field('labels').numpy()
        scores = pred_orig.get_field('scores').numpy()

        # Re-extract avg-pooled RoI features in input-resolution coords (matches base feat pyramid)
        if len(boxes_orig) == 0:
            feats = np.zeros((0, 2048), dtype=np.float32)
        else:
            # Build BoxList in original coords, then resize to input size for RoI pooling
            bxs = BoxList(torch.from_numpy(boxes_orig).float(), (W, H), mode='xyxy').resize((in_w, in_h))
            bxs = bxs.to(model.device)
            base_feat = model.backbone(to_image_list(img_input).to(model.device).tensors)
            roi_feat = model.roi_heads.box.feature_extractor(base_feat, [bxs])  # (N, 2048, 7, 7)
            feats = roi_feat.mean(dim=[2, 3]).cpu().numpy()                      # (N, 2048)
    return {
        'boxes': boxes_orig.astype(np.float32),
        'classes': classes.astype(np.int64),
        'scores': scores.astype(np.float32),
        'feats': feats.astype(np.float32),
    }


def downsample(cv2_img, factor):
    if factor == 1:
        return cv2_img
    H, W = cv2_img.shape[:2]
    return cv2.resize(cv2_img, (max(8, W//factor), max(8, H//factor)), interpolation=cv2.INTER_AREA)


def iou_xyxy(a, b):
    """a, b: (N, 4), (M, 4) → (N, M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1); inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2); inter_y2 = np.minimum(ay2, by2)
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter + 1e-9)


def cos_sim(x, y):
    """row-wise cosine similarity. x: (N, D), y: (N, D) → (N,)"""
    nx = np.linalg.norm(x, axis=1) + 1e-9
    ny = np.linalg.norm(y, axis=1) + 1e-9
    return (x * y).sum(axis=1) / (nx * ny)


def match_and_compare(native, lowres, iou_th=0.5, lowres_scale=1.0):
    """Match native detections to low-res detections via IoU+class. Compute paired stats.
    lowres_scale: scale factor to map low-res boxes back to native coords (== factor used to downsample).
    """
    if len(native['boxes']) == 0 or len(lowres['boxes']) == 0:
        return None
    lr_boxes_native_coord = lowres['boxes'] * lowres_scale
    iou_mat = iou_xyxy(native['boxes'], lr_boxes_native_coord)  # (N_native, N_lowres)

    # Greedy matching: highest IoU pair first, only same class, IoU >= iou_th
    matches = []  # (native_idx, lowres_idx, iou)
    used_n, used_l = set(), set()
    flat = []
    for i in range(len(native['boxes'])):
        for j in range(len(lowres['boxes'])):
            if native['classes'][i] != lowres['classes'][j]:
                continue
            if iou_mat[i, j] < iou_th:
                continue
            flat.append((iou_mat[i, j], i, j))
    flat.sort(reverse=True)
    for iou, i, j in flat:
        if i in used_n or j in used_l: continue
        used_n.add(i); used_l.add(j)
        matches.append((i, j, iou))

    if not matches:
        return {
            'n_native': len(native['boxes']),
            'n_lowres': len(lowres['boxes']),
            'n_matched': 0,
            'match_rate': 0.0,
        }

    n_idx = np.array([m[0] for m in matches])
    l_idx = np.array([m[1] for m in matches])
    ious = np.array([m[2] for m in matches])

    score_delta = native['scores'][n_idx] - lowres['scores'][l_idx]
    feat_cos = cos_sim(native['feats'][n_idx], lowres['feats'][l_idx])

    return {
        'n_native': int(len(native['boxes'])),
        'n_lowres': int(len(lowres['boxes'])),
        'n_matched': int(len(matches)),
        'match_rate': float(len(matches) / len(native['boxes'])),
        'mean_iou': float(ious.mean()),
        'mean_score_native': float(native['scores'][n_idx].mean()),
        'mean_score_lowres': float(lowres['scores'][l_idx].mean()),
        'mean_score_delta': float(score_delta.mean()),
        'mean_feat_cos':    float(feat_cos.mean()),
        'min_feat_cos':     float(feat_cos.min()),
        'p25_feat_cos':     float(np.percentile(feat_cos, 25)),
        'median_feat_cos':  float(np.median(feat_cos)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/frames')
    ap.add_argument('--test_frames_json', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/test_frames.json')
    ap.add_argument('--n_videos', type=int, default=50)
    ap.add_argument('--n_frames_per_video', type=int, default=2)
    ap.add_argument('--factors', nargs='+', type=int, default=[2, 4, 8])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/vinvl_lowres_feature_compare.json')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    with open(args.test_frames_json) as f:
        all_frames = json.load(f)
    by_video = defaultdict(list)
    for fp in all_frames:
        by_video[fp.split('/')[1]].append(fp)
    videos = sorted(by_video.keys())
    random.shuffle(videos)
    sampled_videos = videos[:args.n_videos]

    selected = []
    for v in sampled_videos:
        frames = sorted(by_video[v])
        if len(frames) <= args.n_frames_per_video:
            selected.extend(frames)
        else:
            selected.extend(random.sample(frames, args.n_frames_per_video))
    print(f'Frames: {len(selected)} from {len(sampled_videos)} videos')

    print('Loading VinVL...')
    model, transforms = load_model()

    per_factor_records = {f: [] for f in args.factors}
    for fp in tqdm(selected, desc='frames'):
        full = os.path.join(os.path.dirname(args.frames_root.rstrip('/')), fp)
        if not os.path.exists(full): continue
        cv2_img = cv2.imread(full)
        if cv2_img is None: continue

        # native
        native = detect_with_features(model, transforms, cv2_img)
        if len(native['boxes']) == 0:
            continue
        for fac in args.factors:
            ds_img = downsample(cv2_img, fac)
            try:
                lr = detect_with_features(model, transforms, ds_img)
            except Exception as e:
                print(f'  skip {fp} fac={fac}: {e}')
                continue
            stats = match_and_compare(native, lr, iou_th=0.5, lowres_scale=float(fac))
            if stats:
                stats['frame'] = fp
                stats['native_HxW'] = list(cv2_img.shape[:2])
                stats['lowres_HxW'] = list(ds_img.shape[:2])
                per_factor_records[fac].append(stats)

    # Aggregate
    print('\n========== Aggregate (native vs low-res, IoU≥0.5 same-class match) ==========')
    print(f"{'factor':<6} {'#frames':>8} {'#det_nat':>9} {'#det_lr':>9} {'match%':>8} {'mean_IoU':>9} {'Δscore':>8} {'feat_cos_mean':>14} {'feat_cos_p25':>13} {'feat_cos_min':>13}")
    aggregated = {}
    for fac, rows in per_factor_records.items():
        if not rows:
            print(f'  factor={fac}: empty')
            continue
        n = len(rows)
        agg = {
            'n_frames': n,
            'mean_n_native':  float(np.mean([r['n_native'] for r in rows])),
            'mean_n_lowres':  float(np.mean([r['n_lowres'] for r in rows])),
            'mean_match_rate':float(np.mean([r['match_rate'] for r in rows])),
            'mean_iou':       float(np.mean([r['mean_iou'] for r in rows if 'mean_iou' in r])),
            'mean_score_delta': float(np.mean([r['mean_score_delta'] for r in rows if 'mean_score_delta' in r])),
            'mean_feat_cos':  float(np.mean([r['mean_feat_cos'] for r in rows if 'mean_feat_cos' in r])),
            'p25_feat_cos':   float(np.mean([r['p25_feat_cos'] for r in rows if 'p25_feat_cos' in r])),
            'min_feat_cos':   float(np.mean([r['min_feat_cos'] for r in rows if 'min_feat_cos' in r])),
        }
        aggregated[fac] = agg
        print(f"{fac:<6} {n:>8} {agg['mean_n_native']:>9.2f} {agg['mean_n_lowres']:>9.2f} {100*agg['mean_match_rate']:>7.1f}% {agg['mean_iou']:>9.4f} {agg['mean_score_delta']:>+8.4f} {agg['mean_feat_cos']:>14.4f} {agg['p25_feat_cos']:>13.4f} {agg['min_feat_cos']:>13.4f}")

    out = {'config': vars(args), 'aggregated': aggregated, 'per_frame': per_factor_records}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
