"""
Extract VinVL dets + 2048-dim RoI features for AG test frames at 2× downsample.

Saves to PLA_det_ag_class_lowres2x/ with same {video}/{frame}/dets.npy + feat.npy schema
as the original PLA_det_ag_class cache.

NOTE: bbox 'rect' values are mapped back to ORIGINAL image coordinates (× factor),
so they are directly compatible with the original cache; only 'feat' actually reflects
the lowered resolution.
"""
from __future__ import annotations
import argparse, json, os, random, sys

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
    """Run VinVL on image, return boxes (in image coords), classes, scores, 2048-d avg-pooled feats."""
    H, W = cv2_img.shape[:2]
    img_input = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    img_input, _ = transforms(img_input, target=None)
    in_h, in_w = img_input.shape[1], img_input.shape[2]

    with torch.no_grad():
        prediction = model([img_input.to(model.device)])[0].to('cpu')
        pred_orig = prediction.resize((W, H))
        boxes_orig = pred_orig.bbox.numpy()
        classes = pred_orig.get_field('labels').numpy()
        scores = pred_orig.get_field('scores').numpy()

        if len(boxes_orig) == 0:
            feats = np.zeros((0, 2048), dtype=np.float32)
        else:
            bxs = BoxList(torch.from_numpy(boxes_orig).float(), (W, H), mode='xyxy').resize((in_w, in_h))
            bxs = bxs.to(model.device)
            base_feat = model.backbone(to_image_list(img_input).to(model.device).tensors)
            roi_feat = model.roi_heads.box.feature_extractor(base_feat, [bxs])
            feats = roi_feat.mean(dim=[2, 3]).cpu().numpy()
    return boxes_orig.astype(np.float32), classes.astype(np.int64), scores.astype(np.float32), feats.astype(np.float32)


def map_to_ag(boxes, classes, scores, feats, vinvl_to_ag, scale_to_orig=1.0):
    """Map VinVL detections to AG class space; some VinVL classes map to []. One VinVL class
    may produce multiple AG-class detections (each with its own copy of bbox+feat).
    Returns: dets (object array of dicts), feat_arr (N, 2048).
    All bboxes are scaled by scale_to_orig (e.g., 2.0 if input was 2× downsampled).
    """
    out_dets = []
    out_feats = []
    for i in range(len(boxes)):
        vc = int(classes[i])
        ag_classes = vinvl_to_ag.get(vc, [])
        if not ag_classes:
            continue
        x1, y1, x2, y2 = boxes[i]
        rect = np.array([x1*scale_to_orig, y1*scale_to_orig, x2*scale_to_orig, y2*scale_to_orig], dtype=np.float32)
        for ag_c in ag_classes:
            out_dets.append({
                'rect': rect.copy(),
                'conf': float(scores[i]),
                'class': int(ag_c),
            })
            out_feats.append(feats[i].copy())
    if not out_dets:
        return np.empty((0,), dtype=object), np.zeros((0, 2048), dtype=np.float32)
    arr = np.empty(len(out_dets), dtype=object)
    for i, d in enumerate(out_dets): arr[i] = d
    return arr, np.stack(out_feats, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/frames')
    ap.add_argument('--test_frames_json', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/ag_test_frames.json')
    ap.add_argument('--mapping', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/weak/oi_to_ag_word_map_synset.npy')
    ap.add_argument('--out_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/PLA_det_ag_class_lowres2x')
    ap.add_argument('--factor', type=int, default=2)
    ap.add_argument('--n_videos', type=int, default=-1, help='-1 = all test videos')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shard_idx', type=int, default=0, help='0-based shard index')
    ap.add_argument('--num_shards', type=int, default=1, help='total number of shards')
    ap.add_argument('--skip_existing', action='store_true', default=True)
    ap.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    print(f'Loading test frames from {args.test_frames_json}...')
    with open(args.test_frames_json) as f:
        test_frames = json.load(f)
    # group by video
    from collections import defaultdict
    by_video = defaultdict(list)
    for fp in test_frames:
        by_video[fp.split('/')[0]].append(fp)
    videos = sorted(by_video.keys())
    print(f'  Videos: {len(videos)}, frames: {len(test_frames)}')

    if args.n_videos > 0:
        random.seed(args.seed)
        random.shuffle(videos)
        videos = videos[:args.n_videos]
    if args.num_shards > 1:
        videos = videos[args.shard_idx::args.num_shards]
        print(f'  → Shard {args.shard_idx}/{args.num_shards}: {len(videos)} videos')
    test_frames = sum([sorted(by_video[v]) for v in videos], [])
    if args.n_videos > 0 or args.num_shards > 1:
        print(f'  → Frames to process: {len(test_frames)} from {len(videos)} videos')

    print(f'Loading VinVL mapping from {args.mapping}...')
    vinvl_to_ag = np.load(args.mapping, allow_pickle=True).tolist()
    print(f'  Loaded mapping for {len(vinvl_to_ag)} VinVL classes')

    print('Loading VinVL model...')
    model, transforms = load_model()

    n_done = n_skip = n_fail = 0
    for fp in tqdm(test_frames, desc=f'extract@×{args.factor}'):
        out_dir = os.path.join(args.out_root, fp)
        dets_path = os.path.join(out_dir, 'dets.npy')
        feat_path = os.path.join(out_dir, 'feat.npy')
        if args.skip_existing and os.path.exists(dets_path) and os.path.exists(feat_path):
            n_skip += 1
            continue

        full_path = os.path.join(args.frames_root, fp)
        if not os.path.exists(full_path):
            tqdm.write(f'  fail (missing): {full_path}')
            n_fail += 1
            continue
        cv2_img = cv2.imread(full_path)
        if cv2_img is None:
            n_fail += 1
            continue
        H, W = cv2_img.shape[:2]
        if args.factor != 1:
            ds_img = cv2.resize(cv2_img, (max(8, W//args.factor), max(8, H//args.factor)),
                                interpolation=cv2.INTER_AREA)
        else:
            ds_img = cv2_img

        try:
            boxes, classes, scores, feats = detect_with_features(model, transforms, ds_img)
        except Exception as e:
            tqdm.write(f'  fail {fp}: {e}')
            n_fail += 1
            continue
        dets_arr, feat_arr = map_to_ag(boxes, classes, scores, feats, vinvl_to_ag, scale_to_orig=float(args.factor))

        os.makedirs(out_dir, exist_ok=True)
        np.save(dets_path, dets_arr, allow_pickle=True)
        np.save(feat_path, feat_arr)
        n_done += 1

    print(f'\n[done] extracted={n_done}  skipped={n_skip}  failed={n_fail}  out={args.out_root}')


if __name__ == '__main__':
    main()
