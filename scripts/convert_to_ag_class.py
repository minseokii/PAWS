"""
Convert detections to AG-class-mapped detections with proper feature filtering.

Supports:
1. AG_detection_results (VinVL) → AG_detection_results_ag_class
2. PLA_gdino_* → PLA_gdino_*_ag_class (preserves reliability/match_score)

Usage:
    # Convert VinVL detections
    python convert_to_ag_class.py \
        --src_det /path/to/AG_detection_results \
        --dst /path/to/AG_detection_results_ag_class

    # Convert GDINO detections (needs separate feat source)
    python convert_to_ag_class.py \
        --src_det /path/to/PLA_gdino_12_5 \
        --src_feat /path/to/AG_detection_results \
        --dst /path/to/PLA_gdino_12_5_ag_class
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def convert_detections(src_det_root, src_feat_root, dst_root, mapping):
    """
    Convert detections to AG classes with proper feature filtering.

    Args:
        src_det_root: Source detection folder (dets.npy)
        src_feat_root: Source feature folder (feat.npy) - can be same as src_det_root
        dst_root: Destination folder
        mapping: VinVL class → AG class mapping dict
    """
    videos = sorted(os.listdir(src_det_root))
    total_kept = 0
    total_removed = 0
    total_frames = 0

    for vid in tqdm(videos, desc="Converting"):
        vid_det = os.path.join(src_det_root, vid)
        vid_feat = os.path.join(src_feat_root, vid)

        if not os.path.isdir(vid_det):
            continue

        for frame in sorted(os.listdir(vid_det)):
            frame_det = os.path.join(vid_det, frame)
            frame_feat = os.path.join(vid_feat, frame)

            det_path = os.path.join(frame_det, 'dets.npy')
            feat_path = os.path.join(frame_feat, 'feat.npy')

            if not os.path.exists(det_path):
                continue

            # Load detections
            dets = np.load(det_path, allow_pickle=True)

            # Load features (may be from different path)
            if os.path.exists(feat_path):
                feats = np.load(feat_path, allow_pickle=True)
                has_feats = True
            else:
                feats = None
                has_feats = False

            new_dets = []
            new_feat_indices = []

            for i, d in enumerate(dets):
                vinvl_cls = int(to_numpy(d['class']))
                ag_classes = mapping.get(vinvl_cls, [])

                if len(ag_classes) == 0:
                    total_removed += 1
                    continue

                for ag_cls in ag_classes:
                    new_det = {
                        'rect': to_numpy(d['rect']).astype(np.float32),
                        'conf': float(to_numpy(d['conf'])),
                        'class': ag_cls,
                    }

                    # Preserve GDINO-specific fields if present
                    if 'reliability' in d:
                        new_det['reliability'] = float(d['reliability'])
                    if 'match_score' in d:
                        new_det['match_score'] = float(d['match_score'])

                    new_dets.append(new_det)
                    new_feat_indices.append(i)
                    total_kept += 1

            # Save to destination
            frame_dst = os.path.join(dst_root, vid, frame)
            os.makedirs(frame_dst, exist_ok=True)

            # Save filtered dets
            np.save(os.path.join(frame_dst, 'dets.npy'), new_dets)

            # Save filtered feats (using correct indices!)
            if has_feats and len(new_feat_indices) > 0:
                new_feats = feats[new_feat_indices]
                np.save(os.path.join(frame_dst, 'feat.npy'), new_feats)
            elif has_feats:
                # No detections kept, save empty features
                new_feats = np.empty((0, feats.shape[1]), dtype=feats.dtype)
                np.save(os.path.join(frame_dst, 'feat.npy'), new_feats)

            total_frames += 1

    print(f"\nDone!")
    print(f"  Frames processed: {total_frames}")
    print(f"  Detections kept: {total_kept}")
    print(f"  Detections removed: {total_removed}")


def main():
    parser = argparse.ArgumentParser(description='Convert detections to AG classes')
    parser.add_argument('--src_det', type=str, required=True,
                        help='Source detection folder (dets.npy)')
    parser.add_argument('--src_feat', type=str, default=None,
                        help='Source feature folder (feat.npy). If not specified, uses src_det.')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination folder')
    parser.add_argument('--mapping_path', type=str,
                        default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/weak/oi_to_ag_word_map_synset.npy',
                        help='Path to OI→AG class mapping')
    args = parser.parse_args()

    # If src_feat not specified, use src_det
    if args.src_feat is None:
        args.src_feat = args.src_det

    print(f"Source detections: {args.src_det}")
    print(f"Source features:   {args.src_feat}")
    print(f"Destination:       {args.dst}")
    print()

    # Load mapping
    mapping = np.load(args.mapping_path, allow_pickle=True).item()
    mapped_count = sum(1 for v in mapping.values() if len(v) > 0)
    print(f"Mapping loaded: {len(mapping)} VinVL classes, {mapped_count} mapped to AG")
    print()

    convert_detections(args.src_det, args.src_feat, args.dst, mapping)


if __name__ == '__main__':
    main()
