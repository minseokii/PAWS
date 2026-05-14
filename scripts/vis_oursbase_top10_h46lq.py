"""
Visualize top-10 triplets from Ours (PA/PAM) vs Baseline (no PA/PAM) for a target frame.

For each unique bbox in the top-10 triplets:
  - Cyan (deepskyblue) box if predicted class APPEARS in this frame's GT triplets
  - Pink (hotpink) box otherwise
  - Label: "<cls> <conf>" using the per-detection conf score (2 decimal)

Reference style: visualize/v17_f1/det_base_gtmatch.png
"""
import os, sys, copy, argparse, yaml, pickle
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from easydict import EasyDict as edict
from torchvision.ops import nms as torch_nms

PLA_ROOT = '/SSD1/minseok/WS-DSGG/TRKT/PLA'
sys.path.insert(0, PLA_ROOT)
os.chdir(PLA_ROOT)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='H46LQ.mp4')
    ap.add_argument('--frame', default='000378.png')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--out_dir', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/vis_oursbase_top10')
    ap.add_argument('--ours_remove', nargs='*', default=[], help='class names to drop from OURS viz')
    ap.add_argument('--base_remove', nargs='*', default=[], help='class names to drop from BASELINE viz')
    ap.add_argument('--base_add', nargs='*', default=[], help='class names to add to BASELINE viz from any detection')
    return ap.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.makedirs(args.out_dir, exist_ok=True)

from lib.config import conf, cfg_from_file
from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.assign_pseudo_label import (
    prepare_func, load_feature,
    assign_label_to_proposals_by_dict_for_video, convert_data
)
from lib.sttran import STTran

OURS_YML  = 'configs/pla_stage_2/ours.yml'
OURS_CKPT = 'models/stage2/0212_2321_wk_exact_pabalanced_dist_bce_pamadaptive_unmatched/best_model.tar'
BASE_CKPT = 'models/stage2/0217_1706_wk_exact/best_model.tar'

cfg_from_file(OURS_YML)
gpu_device = torch.device('cuda')
torch.set_num_threads(4)

# === Dataset ===
print('Loading AG test dataset...')
AG_dataset_test = AG(
    mode='test', datasize=conf.datasize, data_path=conf.data_path,
    ws_object_bbox_path=None, remove_one_frame_video=True,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == 'predcls' else True,
)
object_classes = AG_dataset_test.object_classes
attention_relationships = AG_dataset_test.attention_relationships
spatial_relationships = AG_dataset_test.spatial_relationships
contacting_relationships = AG_dataset_test.contacting_relationships
FRAMES_DIR = AG_dataset_test.frames_path
print(f'  {len(AG_dataset_test.video_list)} videos loaded')

# Find target video
target_key = f'{args.video}/{args.frame}'
VIDEO_IDX = None
for i, vlist in enumerate(AG_dataset_test.video_list):
    if vlist and vlist[0].split('/')[0] == args.video:
        VIDEO_IDX = i
        break
if VIDEO_IDX is None:
    raise RuntimeError(f'Video {args.video} not in test set')
frame_names = AG_dataset_test.video_list[VIDEO_IDX]
gt_annotation = AG_dataset_test.gt_annotations[VIDEO_IDX]
real_gt_annotation = AG_dataset_test.real_gt_annotations[VIDEO_IDX]
if target_key not in frame_names:
    raise RuntimeError(f'Frame {target_key} not in this video')
FIV = frame_names.index(target_key)
img_path = os.path.join(FRAMES_DIR, target_key)
print(f'Target: video idx={VIDEO_IDX}, frame idx={FIV}, path={img_path}')

# === Object detector (loads cached features) ===
object_detector_test = detector(
    train=False, object_classes=object_classes, use_SUPPLY=True, conf=conf
).to(device=gpu_device)
object_detector_test.eval()
faset_rcnn_model, transforms = prepare_func()


# === Models ===
def load_sttran(yml_path, ckpt_path, pam_override=None):
    with open(yml_path) as f:
        yml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    model_conf = copy.deepcopy(conf)
    for k, v in yml_cfg.items(): model_conf[k] = v
    if pam_override is not None: model_conf.pam = pam_override

    model = STTran(
        mode=model_conf.mode,
        attention_class_num=len(attention_relationships),
        spatial_class_num=len(spatial_relationships),
        contact_class_num=len(contacting_relationships),
        obj_classes=object_classes,
        enc_layer_num=model_conf.enc_layer,
        dec_layer_num=model_conf.dec_layer,
        transformer_mode=model_conf.transformer_mode,
        is_wks=model_conf.is_wks,
        feat_dim=model_conf.feat_dim,
        obj_dim=model_conf.obj_dim,
        conf=model_conf,
    ).to(gpu_device)

    ckpt = torch.load(ckpt_path, map_location=gpu_device)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing: print(f'  Missing keys: {len(missing)} (sample: {missing[:3]})')
    model.eval()
    return model, model_conf


print('Loading Ours model (pam=True)...')
model_ours, conf_ours = load_sttran(OURS_YML, OURS_CKPT, pam_override=True)
print('Loading Baseline model (pam=False)...')
model_base, conf_base = load_sttran(OURS_YML, BASE_CKPT, pam_override=False)


# === Helpers ===
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1]); a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0.0


def make_entry(object_detector, gt_annotation, real_gt_annotation, frame_names,
               faset_rcnn_model, transforms, use_gt_person=True):
    c = object_detector.conf
    dets_list, feat_list, base_feat_list = load_feature(
        frame_names, c.union_box_feature,
        det_path=c.det_path, feat_path=c.feat_path, is_train=False, load_feat=True)
    if feat_list is None:
        feat_list = object_detector.extract_feat(dets_list, frame_names, faset_rcnn_model, transforms)
    if use_gt_person:
        for fid in range(len(dets_list)):
            if fid >= len(real_gt_annotation): continue
            gt_p = real_gt_annotation[fid][0].get('person_bbox', None)
            if gt_p is None: continue
            gt_box = gt_p[0] if isinstance(gt_p, np.ndarray) and gt_p.ndim == 2 else gt_p
            gt_box = list(gt_box)
            best_iou, best_idx = 0, -1
            for j, det in enumerate(dets_list[fid]):
                if det['class'] == 1:
                    iou = compute_iou(det['rect'], gt_box)
                    if iou > best_iou: best_iou, best_idx = iou, j
            if best_idx >= 0:
                for j, det in enumerate(dets_list[fid]):
                    if det['class'] == 1: det['conf'] = 0.01
                dets_list[fid][best_idx]['conf'] = 999.0

    video_people_det, video_people_feat, video_object_det, video_object_feat = \
        assign_label_to_proposals_by_dict_for_video(
            dets_list, feat_list, False, gt_annotation,
            pseudo_way=c.pseudo_way, match_mode=c.match, conf=c)
    entry = convert_data(
        False, base_feat_list, video_people_det, video_people_feat,
        video_object_det, video_object_feat, gt_annotation, frame_names,
        faset_rcnn_model, transforms, union_box_feature=c.union_box_feature, conf=c)
    return entry


def apply_nms_to_entry(entry, iou_threshold=0.5):
    boxes, labels, scores = entry['boxes'], entry['labels'], entry['scores']
    N = len(labels)
    keep_mask = torch.zeros(N, dtype=torch.bool, device=labels.device)
    keep_mask[labels == 1] = True
    num_frames = int(boxes[:, 0].max().item()) + 1
    for fid in range(num_frames):
        for cls in labels.unique():
            if cls == 1: continue
            cls_mask = (boxes[:, 0] == fid) & (labels == cls)
            if cls_mask.sum() == 0: continue
            cls_idx = torch.where(cls_mask)[0]
            nms_keep = torch_nms(boxes[cls_idx, 1:5], scores[cls_idx], iou_threshold)
            keep_mask[cls_idx[nms_keep]] = True

    keep_idx = torch.where(keep_mask)[0]
    old2new = torch.full((N,), -1, dtype=torch.long, device=labels.device)
    old2new[keep_idx] = torch.arange(len(keep_idx), device=labels.device)
    pair_idx, im_idx = entry['pair_idx'], entry['im_idx']
    valid = (old2new[pair_idx[:, 0]] >= 0) & (old2new[pair_idx[:, 1]] >= 0)

    new_entry = {}
    for k in ['boxes', 'labels', 'scores', 'features', 'distribution']:
        if k in entry: new_entry[k] = entry[k][keep_idx]
    new_entry['pair_idx'] = old2new[pair_idx[valid]].reshape(-1, 2)
    new_entry['im_idx'] = im_idx[valid]
    for k in ['union_feat', 'union_box', 'spatial_masks']:
        if k in entry: new_entry[k] = entry[k][valid]
    for k in ['attention_gt', 'spatial_gt', 'contacting_gt']:
        if k in entry and isinstance(entry[k], list):
            if len(entry[k]) == len(valid):
                new_entry[k] = [entry[k][i] for i in range(len(valid)) if valid[i]]
            else:
                new_entry[k] = entry[k]
    for k in ['human_idx', 'negative_mask', 'rel_gt', 'pred_labels']:
        if k in entry: new_entry[k] = entry[k]
    return new_entry


def extract_triplets(pred, frame_idx, top_k=10):
    """Top-K triplets sorted by predicate-class probability × pa_score (constraint=with)."""
    im_idx = pred['im_idx']
    mask = (im_idx == frame_idx)
    if mask.sum() == 0:
        return []
    pair_idx = pred['pair_idx'][mask]
    boxes = pred['boxes'][:, 1:].cpu()
    labels = pred.get('pred_labels', pred.get('labels')).cpu()
    obj_scores = pred['scores'].cpu()  # per-detection conf

    attn_dist = F.softmax(pred['attention_distribution'][mask], dim=1).cpu()
    spa_dist = pred['spatial_distribution'][mask].cpu()
    con_dist = pred['contacting_distribution'][mask].cpu()
    pa_scores = None
    if 'pair_affinity' in pred:
        pa_scores = torch.sigmoid(pred['pair_affinity'][mask]).cpu().squeeze(-1)

    triplets = []
    for k in range(pair_idx.shape[0]):
        sid, oid = pair_idx[k, 0].item(), pair_idx[k, 1].item()
        pa = pa_scores[k].item() if pa_scores is not None else 1.0
        attn_idx = attn_dist[k].argmax().item()
        triplets.append({
            'subj_idx': sid, 'obj_idx': oid, 'rel_type': 'attention',
            'rel_cls': attention_relationships[attn_idx],
            'score': attn_dist[k, attn_idx].item() * pa, 'pa': pa})
        sp_idx = spa_dist[k].argmax().item()
        triplets.append({
            'subj_idx': sid, 'obj_idx': oid, 'rel_type': 'spatial',
            'rel_cls': spatial_relationships[sp_idx],
            'score': spa_dist[k, sp_idx].item() * pa, 'pa': pa})
        co_idx = con_dist[k].argmax().item()
        triplets.append({
            'subj_idx': sid, 'obj_idx': oid, 'rel_type': 'contacting',
            'rel_cls': contacting_relationships[co_idx],
            'score': con_dist[k, co_idx].item() * pa, 'pa': pa})

    triplets.sort(key=lambda x: x['score'], reverse=True)
    triplets = triplets[:top_k]

    # Attach metadata (box, class, conf)
    out = []
    for t in triplets:
        sid, oid = t['subj_idx'], t['obj_idx']
        out.append({
            **t,
            'subj_box': boxes[sid].tolist(),
            'subj_cls': object_classes[labels[sid].item()],
            'subj_cls_idx': int(labels[sid].item()),
            'subj_conf': float(obj_scores[sid].item()),
            'obj_box': boxes[oid].tolist(),
            'obj_cls': object_classes[labels[oid].item()],
            'obj_cls_idx': int(labels[oid].item()),
            'obj_conf': float(obj_scores[oid].item()),
        })
    return out


def get_gt_object_classes(real_gt_frame):
    """Return set of GT object class names appearing in this frame."""
    classes = set()
    if not real_gt_frame: return classes
    classes.add('person')
    for o in real_gt_frame[1:]:
        cls_idx = o.get('class')
        if cls_idx is not None and cls_idx < len(object_classes):
            classes.add(object_classes[cls_idx])
    return classes


# === Run inference ===
print('\nRunning inference...')
with torch.no_grad():
    entry = make_entry(object_detector_test, gt_annotation, real_gt_annotation,
                       frame_names, faset_rcnn_model, transforms, use_gt_person=True)
entry = apply_nms_to_entry(entry, iou_threshold=0.5)

with torch.no_grad():
    pred_ours = model_ours(copy.deepcopy(entry))
    pred_base = model_base(copy.deepcopy(entry))

triplets_ours = extract_triplets(pred_ours, FIV, top_k=args.top_k)
triplets_base = extract_triplets(pred_base, FIV, top_k=args.top_k)
gt_classes = get_gt_object_classes(real_gt_annotation[FIV])
print(f'\nGT classes in this frame: {sorted(gt_classes)}')

# === Print top-10 triplets ===
def print_top(triplets, label):
    print(f'\n--- {label} top-{len(triplets)} triplets ---')
    for i, t in enumerate(triplets, 1):
        print(f"  #{i:>2d} ({t['score']:.3f})  {t['subj_cls']}({t['subj_conf']:.2f}) "
              f"--[{t['rel_type'][0].upper()}/{t['rel_cls']}]--> "
              f"{t['obj_cls']}({t['obj_conf']:.2f})")

print_top(triplets_base, 'BASELINE')
print_top(triplets_ours, 'OURS')


# === Visualize ===
def collect_unique_bboxes(triplets):
    """Map cls_name → {box, conf, in_gt}. If multiple boxes per class, keep the one
    used in the highest-ranked triplet."""
    seen = {}  # key = (cls_name, rounded_box) → first-seen entry
    for rank, t in enumerate(triplets):
        for prefix in ('subj', 'obj'):
            cls = t[f'{prefix}_cls']
            box = tuple(round(v, 1) for v in t[f'{prefix}_box'])
            conf = t[f'{prefix}_conf']
            key = (cls, box)
            if key not in seen:
                seen[key] = {'cls': cls, 'box': list(t[f'{prefix}_box']), 'conf': conf, 'rank': rank}
    return list(seen.values())


def find_box_in_frame(entry, frame_idx, cls_name):
    """Return (box, conf) for given class on given frame, picking highest-conf box."""
    boxes = entry['boxes'].cpu()
    labels = entry['labels'].cpu()
    scores = entry['scores'].cpu()
    if cls_name not in object_classes: return None
    cls_idx = object_classes.index(cls_name)
    mask = (boxes[:, 0] == frame_idx) & (labels == cls_idx)
    idx = torch.where(mask)[0]
    if len(idx) == 0: return None
    best = idx[scores[idx].argmax().item()]
    return boxes[best, 1:5].tolist(), float(scores[best].item())


UPSCALE = 3  # upscale output image for crisp text rendering
DISPLAY_NAME = {'cup/glass/bottle': 'glass'}  # short display labels

def render(img_path, triplets, gt_classes, save_path,
           remove_classes=(), add_classes_from_entry=(), entry=None, frame_idx=None,
           override_person_conf=None):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] * UPSCALE, img.shape[0] * UPSCALE), interpolation=cv2.INTER_CUBIC)
    H, W = img.shape[:2]
    items = collect_unique_bboxes(triplets)
    # Drop requested
    if remove_classes:
        items = [it for it in items if it['cls'] not in remove_classes]
    # Add requested classes from entry (any detected box of that class on this frame)
    if add_classes_from_entry and entry is not None and frame_idx is not None:
        for cls in add_classes_from_entry:
            res = find_box_in_frame(entry, frame_idx, cls)
            if res is None: continue
            box, conf = res
            items.append({'cls': cls, 'box': box, 'conf': conf, 'rank': 99})
    # Override person conf if provided (replace 999 with real VinVL conf)
    if override_person_conf is not None:
        for it in items:
            if it['cls'] == 'person':
                it['conf'] = override_person_conf
    # cyan if class in GT, pink otherwise. BGR.
    CYAN  = (255, 192, 0)        # deepskyblue
    PINK  = (180, 105, 255)      # hotpink
    print(f'  [render {os.path.basename(save_path)}] gt_classes={sorted(gt_classes)}')
    for it in items:
        in_gt = it['cls'] in gt_classes
        color = CYAN if in_gt else PINK
        print(f'    {it["cls"]:20s} conf={it["conf"]:.2f}  in_gt={in_gt}  -> {"CYAN" if in_gt else "PINK"}')
        x1, y1, x2, y2 = [int(round(v * UPSCALE)) for v in it['box']]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 * UPSCALE)
        disp = DISPLAY_NAME.get(it['cls'], it['cls'])
        label = f"{disp} {it['conf']:.2f}"
        # filled label background — bold text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.7
        thick = 4  # bold
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
        pad = 10
        bg_w = tw + pad*2
        bg_h = th + baseline + pad*2
        # keep label box fully inside the image (shift left/down if it would overflow)
        bg_x1 = min(max(0, x1), max(0, W - bg_w))
        bg_y1 = y1 - bg_h
        if bg_y1 < 0:
            bg_y1 = min(y1, H - bg_h)  # put it below the top edge instead
        bg_y1 = max(0, bg_y1)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x1 + bg_w, bg_y1 + bg_h), color, -1)
        cv2.putText(img, label, (bg_x1 + pad, bg_y1 + pad + th),
                    font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.imwrite(save_path, img)
    print(f'  Saved {save_path}')


# Recover the original VinVL person conf for this frame (force-override set it to 999)
def get_real_person_conf():
    raw_dets = np.load(os.path.join(conf.det_path, target_key, 'dets.npy'), allow_pickle=True)
    gt_p = real_gt_annotation[FIV][0].get('person_bbox', None)
    if gt_p is None: return None
    gt_box = gt_p[0] if isinstance(gt_p, np.ndarray) and gt_p.ndim == 2 else gt_p
    gt_box = list(gt_box)
    best_iou, best_conf = 0.0, None
    for d in raw_dets:
        if int(d['class']) != 1: continue
        iou_val = compute_iou(d['rect'].tolist() if hasattr(d['rect'], 'tolist') else list(d['rect']), gt_box)
        if iou_val > best_iou:
            best_iou = iou_val
            best_conf = float(d['conf'])
    return best_conf
person_conf = get_real_person_conf()
print(f'\nReal VinVL person conf (GT-matched): {person_conf}')

tag = f'{args.video.replace(".mp4","")}_{args.frame.replace(".png","")}'
render(img_path, triplets_base, gt_classes,
       os.path.join(args.out_dir, f'{tag}_baseline_top{args.top_k}.png'),
       remove_classes=args.base_remove, add_classes_from_entry=args.base_add,
       entry=entry, frame_idx=FIV, override_person_conf=person_conf)
render(img_path, triplets_ours, gt_classes,
       os.path.join(args.out_dir, f'{tag}_ours_top{args.top_k}.png'),
       remove_classes=args.ours_remove,
       entry=entry, frame_idx=FIV, override_person_conf=person_conf)

# Save metadata
import json
out_meta = {
    'video': args.video, 'frame': args.frame,
    'gt_classes': sorted(gt_classes),
    'baseline_topK': triplets_base,
    'ours_topK': triplets_ours,
}
with open(os.path.join(args.out_dir, f'{tag}_meta.json'), 'w') as f:
    json.dump(out_meta, f, indent=2)
print(f'\nDone. Output dir: {args.out_dir}')
