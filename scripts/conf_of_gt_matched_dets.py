"""
Find detections that:
  - Match a GT bbox: IoU >= 0.5 with a GT box of the same class
  - Have low VinVL confidence

This is the "TP-but-low-conf" cohort = occlusion-suspect proxy.

Why: A correctly-classified detection (matches GT class + IoU) with low conf means
the object IS there and was correctly recognized, but its features were degraded
(partial visibility, blur, occlusion) → the most direct proxy for "occluded but
still detected" objects.
"""
import argparse, json, os, pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    inter = iw*ih
    aa = max(0, (a[2]-a[0])*(a[3]-a[1]))
    bb = max(0, (b[2]-b[0])*(b[3]-b[1]))
    return inter / (aa+bb-inter+1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/PLA_det_ag_class')
    ap.add_argument('--ag_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome')
    ap.add_argument('--test_frames', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/ag_test_frames.json')
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--out', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/conf_of_gt_matched_dets.json')
    args = ap.parse_args()

    print('Loading person/object GT pickles...')
    with open(os.path.join(args.ag_root, 'annotations/person_bbox.pkl'), 'rb') as f:
        person_bbox = pickle.load(f)
    with open(os.path.join(args.ag_root, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
        object_bbox = pickle.load(f)

    # AG class index
    classes = ['__background__']
    with open(os.path.join(args.ag_root, 'annotations/object_classes.txt')) as f:
        for line in f:
            line = line.strip()
            if line: classes.append(line.replace('/', ''))
    cls2idx = {c: i for i, c in enumerate(classes)}
    print(f'AG classes: {len(classes)} (incl. background, person=class 1)')

    # ag_test_frames.json items look like '<video>/<frame>.png'
    with open(args.test_frames) as f:
        test_frames = json.load(f)

    # AG keys are usually '<video>/<frame>.png'
    print('Sample person_bbox keys:', list(person_bbox.keys())[:3])
    print('Sample object_bbox keys:', list(object_bbox.keys())[:3])

    # Process
    matched_records = []  # {conf, class, iou, box_area_frac, n_gt_in_frame}
    n_frames_seen = n_frames_with_gt = 0
    n_dets_total = n_dets_matched = 0

    for fp in tqdm(test_frames, desc='match'):
        # Cache path
        dets_path = os.path.join(args.cache_root, fp, 'dets.npy')
        if not os.path.exists(dets_path): continue
        dets = np.load(dets_path, allow_pickle=True)
        if len(dets) == 0: continue
        n_frames_seen += 1
        n_dets_total += len(dets)

        # Get GT for this frame
        # Person GT: person_bbox[fp] = {'bbox': np.array([x1,y1,x2,y2]), ...}; class=1
        gt_boxes = []  # list of (np.array(rect), class_idx)
        if fp in person_bbox:
            pb = person_bbox[fp]
            if isinstance(pb, dict) and 'bbox' in pb:
                bb = pb['bbox']
                if bb is not None and np.size(bb) >= 4:
                    bb = np.asarray(bb).flatten()[:4]
                    gt_boxes.append((bb, cls2idx['person']))
        # Object GT
        if fp in object_bbox:
            for o in object_bbox[fp]:
                if not o.get('visible', True): continue  # skip invisible labels
                bb = o.get('bbox')
                cls = o.get('class')
                if bb is None or cls is None: continue
                bb = np.asarray(bb).flatten()
                if bb.size < 4: continue
                # AG bbox format: [x1, y1, w, h]  (need to confirm)
                # Try both interpretations and use whichever works; will fix below
                gt_boxes.append((bb[:4], cls2idx.get(cls.replace('/', ''), -1)))

        if not gt_boxes: continue
        n_frames_with_gt += 1

        # Convert AG GT boxes to xyxy if they look like xywh
        # Detector boxes are stored as xyxy already.
        # AG object_bbox format: [x1, y1, w, h] in original image coords
        # AG person_bbox format: usually xyxy
        # Heuristic: if x2 < x1 or y2 < y1 in xyxy interpretation, treat as xywh
        gt_xyxy = []
        for bb, cls in gt_boxes:
            x1, y1, a, b = bb
            # Try as xywh: x2 = x1+w, y2 = y1+h
            if a > 0 and b > 0:
                if a < x1 or b < y1:  # raw 'a' looks smaller — likely w
                    # likely xywh
                    gt_xyxy.append((np.array([x1, y1, x1+a, y1+b], dtype=np.float32), cls))
                else:
                    # could be xyxy already
                    gt_xyxy.append((np.array([x1, y1, a, b], dtype=np.float32), cls))
            else:
                continue

        # For each det, find matching GT (max IoU among same-class GT, must be >= iou_thr)
        for d in dets:
            d_rect = np.asarray(d['rect']).flatten()[:4]
            d_cls = int(d['class'])
            d_conf = float(d['conf'])
            best_iou = 0.0
            for (gb, gc) in gt_xyxy:
                if gc != d_cls: continue
                u = iou(d_rect, gb)
                if u > best_iou:
                    best_iou = u
            if best_iou >= args.iou_thr:
                # box area as fraction of typical 480p frame (480*270 = 129600)
                area = max(0, (d_rect[2]-d_rect[0])*(d_rect[3]-d_rect[1]))
                matched_records.append({
                    'conf': d_conf,
                    'class': d_cls,
                    'iou': best_iou,
                    'area': float(area),
                })
                n_dets_matched += 1

    print(f'\nFrames seen: {n_frames_seen}, with usable GT: {n_frames_with_gt}')
    print(f'Total detections: {n_dets_total}, GT-matched (IoU>={args.iou_thr}, same class): {n_dets_matched}')
    print(f'Match rate: {100*n_dets_matched/n_dets_total:.2f}%')

    if not matched_records:
        print('No matched records — abort.')
        return

    confs = np.array([r['conf'] for r in matched_records], dtype=np.float32)
    print('\n=== Conf distribution of GT-matched detections ===')
    print(f'  mean={confs.mean():.4f}  median={np.median(confs):.4f}  std={confs.std():.4f}')
    print(f'  min={confs.min():.4f}  max={confs.max():.4f}')

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(confs, pcts)
    for p, v in zip(pcts, pct_vals):
        print(f'  p{p:>2d} = {v:.4f}')

    bins = np.arange(0.0, 1.05, 0.05)
    hist, _ = np.histogram(confs, bins=bins)
    print('\nHistogram (matched dets):')
    cum = 0
    total = len(confs)
    for i in range(len(hist)):
        cum += hist[i]
        print(f'  [{bins[i]:.2f}, {bins[i+1]:.2f})  n={hist[i]:>7d}  ({100*hist[i]/total:5.2f}%)  cum={100*cum/total:5.2f}%')

    print('\nFraction of GT-matched dets at thresholds:')
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        below = (confs < t).sum()
        print(f'  conf < {t:.2f}: {below} ({100*below/total:5.2f}%)')

    out = {
        'iou_thr': args.iou_thr,
        'n_frames_seen': n_frames_seen,
        'n_frames_with_gt': n_frames_with_gt,
        'n_dets_total': n_dets_total,
        'n_dets_matched': n_dets_matched,
        'match_rate': float(n_dets_matched/n_dets_total),
        'matched_conf_mean': float(confs.mean()),
        'matched_conf_median': float(np.median(confs)),
        'matched_conf_std': float(confs.std()),
        'matched_percentiles': {f'p{p}': float(v) for p, v in zip(pcts, pct_vals)},
        'histogram_bins': bins.tolist(),
        'histogram_counts': hist.tolist(),
        'fraction_below_thresh': {str(t): float((confs < t).sum() / total) for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
