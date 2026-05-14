"""
Compute VinVL detection confidence distribution on AG test cache.

Reads dets.npy from PLA_det_ag_class/{video}/{frame}/dets.npy and aggregates
the 'conf' field across all detections.
"""
import argparse, json, os, glob
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_root', default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/PLA_det_ag_class')
    ap.add_argument('--test_frames', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/ag_test_frames.json')
    ap.add_argument('--out', default='/SSD1/minseok/WS-DSGG/TRKT/PLA/scripts/_logs/conf_distribution.json')
    args = ap.parse_args()

    with open(args.test_frames) as f:
        test_frames = json.load(f)
    print(f'Test frames: {len(test_frames)}')

    confs_all = []
    confs_per_class = {}
    n_files = 0
    n_missing = 0
    for fp in tqdm(test_frames, desc='reading'):
        dets_path = os.path.join(args.cache_root, fp, 'dets.npy')
        if not os.path.exists(dets_path):
            n_missing += 1
            continue
        dets = np.load(dets_path, allow_pickle=True)
        n_files += 1
        for d in dets:
            c = float(d['conf'])
            cls = int(d['class'])
            confs_all.append(c)
            confs_per_class.setdefault(cls, []).append(c)

    confs_all = np.array(confs_all, dtype=np.float32)
    print(f'\nFiles read: {n_files}, missing: {n_missing}')
    print(f'Total detections: {len(confs_all)}')
    print(f'  mean={confs_all.mean():.4f}  median={np.median(confs_all):.4f}  std={confs_all.std():.4f}')
    print(f'  min={confs_all.min():.4f}  max={confs_all.max():.4f}')

    print('\nPercentiles:')
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(confs_all, pcts)
    for p, v in zip(pcts, pct_vals):
        print(f'  p{p:>2d} = {v:.4f}')

    print('\nHistogram (bins of 0.05):')
    bins = np.arange(0.0, 1.05, 0.05)
    hist, _ = np.histogram(confs_all, bins=bins)
    cum = 0
    total = len(confs_all)
    for i in range(len(hist)):
        cum += hist[i]
        print(f'  [{bins[i]:.2f}, {bins[i+1]:.2f})  n={hist[i]:>8d}  ({100*hist[i]/total:5.2f}%)  cum_below_upper={100*cum/total:5.2f}%')

    print('\nFraction at common thresholds:')
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        below = (confs_all < t).sum()
        above = (confs_all >= t).sum()
        print(f'  conf < {t}: {below} ({100*below/total:5.2f}%)  |  conf >= {t}: {above} ({100*above/total:5.2f}%)')

    out = {
        'n_detections_total': int(len(confs_all)),
        'n_frames_read': n_files,
        'n_frames_missing': n_missing,
        'mean': float(confs_all.mean()),
        'median': float(np.median(confs_all)),
        'std': float(confs_all.std()),
        'percentiles': {f'p{p}': float(v) for p, v in zip(pcts, pct_vals)},
        'histogram_bins': bins.tolist(),
        'histogram_counts': hist.tolist(),
        'fraction_below_threshold': {str(t): float((confs_all < t).sum() / total) for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {args.out}')


if __name__ == '__main__':
    main()
