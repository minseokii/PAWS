"""
Visualize cases where:
  - Detection class is NOT person
  - Detection IoU >= 0.5 with a same-class GT bbox
  - VinVL conf < 0.3

These are 'occlusion-suspect' samples. Save annotated images:
  - GT bbox (green) with class label
  - Detection bbox (red) with class + conf
  - Person GT bbox (blue, for context)
"""
import argparse, json, os, pickle, random
import numpy as np
import cv2
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
    ap.add_argument('--frames_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/frames')
    ap.add_argument('--test_frames', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/ag_test_frames.json')
    ap.add_argument('--conf_max', type=float, default=0.30)
    ap.add_argument('--iou_min', type=float, default=0.5)
    ap.add_argument('--n_samples', type=int, default=24)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out_dir', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/vis_occlusion_proxy')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # AG classes
    classes = ['__background__']
    with open(os.path.join(args.ag_root, 'annotations/object_classes.txt')) as f:
        for line in f:
            line = line.strip()
            if line: classes.append(line.replace('/', ''))
    cls2idx = {c: i for i, c in enumerate(classes)}
    PERSON = cls2idx['person']
    print(f'AG classes: {len(classes)}, person idx={PERSON}')

    print('Loading GT pickles...')
    with open(os.path.join(args.ag_root, 'annotations/person_bbox.pkl'), 'rb') as f:
        person_bbox = pickle.load(f)
    with open(os.path.join(args.ag_root, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
        object_bbox = pickle.load(f)

    with open(args.test_frames) as f:
        test_frames = json.load(f)

    random.seed(args.seed)
    random.shuffle(test_frames)

    candidates = []  # list of dicts: frame, det, matched_gt
    print(f'Searching for low-conf object detections with GT match (target {args.n_samples * 5})...')

    for fp in tqdm(test_frames):
        if len(candidates) >= args.n_samples * 5: break

        dets_path = os.path.join(args.cache_root, fp, 'dets.npy')
        if not os.path.exists(dets_path): continue
        dets = np.load(dets_path, allow_pickle=True)
        if len(dets) == 0: continue

        # Build GT object boxes (skip person, only objects with visible bbox)
        if fp not in object_bbox: continue
        gt_obj_xyxy = []
        for o in object_bbox[fp]:
            if not o.get('visible', True): continue
            bb = o.get('bbox')
            cls = o.get('class')
            if bb is None or cls is None: continue
            bb = np.asarray(bb).flatten()
            if bb.size < 4: continue
            x1, y1, w, h = bb[:4]
            if w <= 0 or h <= 0: continue
            cls_idx = cls2idx.get(cls.replace('/', ''), -1)
            if cls_idx < 0 or cls_idx == PERSON: continue
            gt_obj_xyxy.append({
                'rect': np.array([x1, y1, x1+w, y1+h], dtype=np.float32),
                'class': cls_idx,
                'class_name': cls.replace('/', ''),
            })
        if not gt_obj_xyxy: continue

        # Person GT (for context only)
        person_xyxy = None
        if fp in person_bbox:
            pb = person_bbox[fp]
            if isinstance(pb, dict) and pb.get('bbox') is not None:
                pb_arr = np.asarray(pb['bbox']).flatten()[:4]
                if pb_arr.size == 4:
                    person_xyxy = pb_arr.astype(np.float32)

        # For each detection: object class only, conf < threshold
        for d in dets:
            d_cls = int(d['class'])
            d_conf = float(d['conf'])
            if d_cls == PERSON: continue
            if d_conf >= args.conf_max: continue
            d_rect = np.asarray(d['rect']).flatten()[:4].astype(np.float32)
            # Find best GT match (same class)
            best_iou = 0.0
            best_gt = None
            for gt in gt_obj_xyxy:
                if gt['class'] != d_cls: continue
                u = iou(d_rect, gt['rect'])
                if u > best_iou:
                    best_iou = u
                    best_gt = gt
            if best_iou >= args.iou_min and best_gt is not None:
                candidates.append({
                    'frame': fp,
                    'det_rect': d_rect.tolist(),
                    'det_class': d_cls,
                    'det_class_name': classes[d_cls],
                    'det_conf': d_conf,
                    'gt_rect': best_gt['rect'].tolist(),
                    'gt_class_name': best_gt['class_name'],
                    'iou': best_iou,
                    'person_rect': person_xyxy.tolist() if person_xyxy is not None else None,
                })

    print(f'\nFound {len(candidates)} candidates. Sampling {args.n_samples} for visualization.')
    if not candidates:
        print('No candidates — abort.')
        return

    random.seed(args.seed + 1)
    # Diversify: avoid two from same video where possible
    by_video = {}
    for c in candidates:
        v = c['frame'].split('/')[0]
        by_video.setdefault(v, []).append(c)
    videos_shuf = list(by_video.keys())
    random.shuffle(videos_shuf)
    selected = []
    for v in videos_shuf:
        if len(selected) >= args.n_samples: break
        selected.append(random.choice(by_video[v]))

    # Render
    out_meta = []
    for i, s in enumerate(selected):
        img_path = os.path.join(args.frames_root, s['frame'])
        if not os.path.exists(img_path):
            print(f'  missing: {img_path}'); continue
        img = cv2.imread(img_path)
        if img is None: continue
        H, W = img.shape[:2]

        # Person bbox (blue)
        if s.get('person_rect'):
            x1, y1, x2, y2 = [int(round(v)) for v in s['person_rect']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 128, 0), 1)
            cv2.putText(img, 'person GT', (x1+2, max(12, y1-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1, cv2.LINE_AA)

        # GT (green)
        x1, y1, x2, y2 = [int(round(v)) for v in s['gt_rect']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"GT: {s['gt_class_name']}", (x1+2, min(H-3, y2+13)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)

        # Det (red)
        x1, y1, x2, y2 = [int(round(v)) for v in s['det_rect']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"DET: {s['det_class_name']} c={s['det_conf']:.2f} iou={s['iou']:.2f}",
                    (x1+2, max(13, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Header
        header = f"{i:02d}: {s['frame']}  | {s['det_class_name']} conf={s['det_conf']:.2f}"
        # add white strip on top with header
        strip = np.full((22, W, 3), 255, dtype=np.uint8)
        cv2.putText(strip, header, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        out = np.vstack([strip, img])

        save_name = f"{i:02d}_{s['frame'].replace('/', '_').replace('.png','')}_{s['det_class_name']}_c{s['det_conf']:.2f}.png"
        save_path = os.path.join(args.out_dir, save_name)
        cv2.imwrite(save_path, out)
        out_meta.append({**s, 'saved': save_name})

    # Build a 4xN grid montage for quick review
    if out_meta:
        # Read back saved images to compose montage
        imgs = []
        for m in out_meta:
            p = os.path.join(args.out_dir, m['saved'])
            im = cv2.imread(p)
            if im is None: continue
            # Resize to common width
            target_w = 480
            scale = target_w / im.shape[1]
            target_h = int(im.shape[0] * scale)
            im = cv2.resize(im, (target_w, target_h))
            imgs.append(im)
        if imgs:
            # pad to common height
            max_h = max(im.shape[0] for im in imgs)
            imgs_p = [np.pad(im, ((0, max_h - im.shape[0]), (0, 0), (0, 0)), constant_values=255) for im in imgs]
            ncol = 4
            rows = []
            for r in range(0, len(imgs_p), ncol):
                row = imgs_p[r:r+ncol]
                while len(row) < ncol: row.append(np.full_like(imgs_p[0], 255))
                rows.append(np.hstack(row))
            grid = np.vstack(rows)
            cv2.imwrite(os.path.join(args.out_dir, 'GRID_overview.png'), grid)
            print(f'Grid → {os.path.join(args.out_dir, "GRID_overview.png")}')

    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump({'config': vars(args), 'candidates_total': len(candidates), 'selected': out_meta}, f, indent=2)
    print(f'Saved {len(out_meta)} samples to {args.out_dir}')


if __name__ == '__main__':
    main()
