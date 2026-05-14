"""
Smoke test: How does VinVL detection degrade as input resolution drops?

Picks N random AG test videos × M frames, runs VinVL on:
  - native (e.g., 480p Charades source)
  - downsample × {2, 4, 8}  (240p, 120p, 60p)

Reports per-resolution: #detections, mean score, top class distribution.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import Counter, defaultdict

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
    return model, transforms, cfg


def detect(model, transforms, cv2_img):
    img_input = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)
    with torch.no_grad():
        prediction = model(img_input)[0].to('cpu')
    H, W = cv2_img.shape[:2]
    prediction = prediction.resize((W, H))
    return {
        'boxes':   prediction.bbox.numpy(),
        'classes': prediction.get_field('labels').numpy(),
        'scores':  prediction.get_field('scores').numpy(),
    }


def downsample(cv2_img, factor):
    """Downsample then keep at smaller resolution (VinVL preprocess will then upscale)."""
    if factor == 1:
        return cv2_img
    H, W = cv2_img.shape[:2]
    new_w = max(8, W // factor)
    new_h = max(8, H // factor)
    return cv2.resize(cv2_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def summarize(dets, score_thresh=0.2):
    """Return a dict of stats for one detection result."""
    if dets is None:
        return None
    s = dets['scores']
    keep = s >= score_thresh
    s_keep = s[keep]
    cls_keep = dets['classes'][keep]
    return {
        'n_total':     int(len(s)),
        'n_above_thr': int(keep.sum()),
        'mean_score':  float(s.mean()) if len(s) else 0.0,
        'max_score':   float(s.max()) if len(s) else 0.0,
        'mean_score_kept': float(s_keep.mean()) if keep.sum() else 0.0,
        'unique_classes':  int(len(set(cls_keep.tolist()))),
        'class_dist':  Counter(cls_keep.tolist()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/frames')
    ap.add_argument('--test_frames_json', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/test_frames.json')
    ap.add_argument('--n_videos', type=int, default=50)
    ap.add_argument('--n_frames_per_video', type=int, default=2)
    ap.add_argument('--factors', nargs='+', type=int, default=[1, 2, 4, 8])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/vinvl_lowres_smoke.json')
    ap.add_argument('--score_thresh', type=float, default=0.2)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print('Loading test_frames.json...')
    with open(args.test_frames_json) as f:
        all_test_frames = json.load(f)
    # group by video
    by_video = defaultdict(list)
    for fp in all_test_frames:
        vid = fp.split('/')[1]  # 'frames/<video>/<frame>.png'
        by_video[vid].append(fp)
    print(f'  Test videos: {len(by_video)}')

    videos = sorted(by_video.keys())
    random.shuffle(videos)
    sampled_videos = videos[:args.n_videos]

    selected_frames = []
    for v in sampled_videos:
        frames = sorted(by_video[v])
        if not frames: continue
        if len(frames) <= args.n_frames_per_video:
            selected_frames.extend(frames)
        else:
            selected_frames.extend(random.sample(frames, args.n_frames_per_video))
    print(f'  Selected {len(selected_frames)} frames from {len(sampled_videos)} videos')

    print('Loading VinVL...')
    model, transforms, _ = load_model()

    # Per-factor accumulators
    per_factor = {f: [] for f in args.factors}
    per_frame_records = []  # for paired comparison

    for fp in tqdm(selected_frames, desc='frames'):
        full_path = os.path.join(os.path.dirname(args.frames_root.rstrip('/')), fp) if not fp.startswith('frames/') else os.path.join(os.path.dirname(args.frames_root.rstrip('/')), fp)
        # actually frames_root is .../frames so combine with the relative path properly
        if fp.startswith('frames/'):
            full_path = os.path.join(os.path.dirname(args.frames_root.rstrip('/')), fp)
        else:
            full_path = os.path.join(args.frames_root, fp)
        if not os.path.exists(full_path):
            continue
        cv2_img = cv2.imread(full_path)
        if cv2_img is None: continue
        H0, W0 = cv2_img.shape[:2]

        rec = {'frame': fp, 'native_HxW': [H0, W0], 'by_factor': {}}
        for fac in args.factors:
            ds = downsample(cv2_img, fac)
            try:
                d = detect(model, transforms, ds)
            except Exception as e:
                print(f'   skip {fp} factor={fac}: {e}')
                continue
            stats = summarize(d, score_thresh=args.score_thresh)
            stats['input_HxW'] = list(ds.shape[:2])
            rec['by_factor'][fac] = stats
            per_factor[fac].append(stats)
        per_frame_records.append(rec)

    # Aggregate
    print('\n========== Aggregate (n_frames per factor) ==========')
    print(f"{'factor':<8} {'input_h_med':>11} {'#frames':>9} {'#det_total':>11} {'#det_kept':>11} {'mean_score':>11} {'mean_score_kept':>16} {'unique_cls':>12}")
    aggregated = {}
    for fac in args.factors:
        rows = per_factor[fac]
        if not rows:
            print(f'  factor={fac}: no data')
            continue
        n_total = np.mean([r['n_total'] for r in rows])
        n_kept  = np.mean([r['n_above_thr'] for r in rows])
        m_score = np.mean([r['mean_score'] for r in rows])
        m_score_kept = np.mean([r['mean_score_kept'] for r in rows])
        u_cls   = np.mean([r['unique_classes'] for r in rows])
        # input height median (after downsample)
        h_med = int(np.median([per_frame_records[i]['by_factor'].get(fac, {}).get('input_HxW', [0,0])[0] for i in range(len(per_frame_records)) if fac in per_frame_records[i]['by_factor']]))
        print(f"{fac:<8} {h_med:>11d} {len(rows):>9d} {n_total:>11.2f} {n_kept:>11.2f} {m_score:>11.4f} {m_score_kept:>16.4f} {u_cls:>12.2f}")
        aggregated[fac] = {
            'n_frames': len(rows),
            'input_h_median': h_med,
            'mean_n_total':  float(n_total),
            'mean_n_kept':   float(n_kept),
            'mean_score':    float(m_score),
            'mean_score_kept': float(m_score_kept),
            'mean_unique_classes': float(u_cls),
        }

    # Save
    out = {
        'config': {k:v for k,v in vars(args).items()},
        'aggregated': aggregated,
        'per_frame': per_frame_records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Make Counters JSON serializable
    for r in per_frame_records:
        for fac, s in r.get('by_factor', {}).items():
            s['class_dist'] = dict(s['class_dist'])
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
