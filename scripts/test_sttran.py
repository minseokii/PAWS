import os
import argparse
import json
from lib.config import conf, cfg_from_file
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/pla_stage_2/sttran_paws.yml', type=str)
parser.add_argument('--save_result', type=str, default=None,
                    help='Save per-frame predictions/GT/recall to pkl (e.g., ours_result.pkl)')
args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
print('The results saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print(conf)

os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_id)


import pickle
import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
torch.set_num_threads(2)

from dataloader.action_genome import AG, cuda_collate_fn

from lib.evaluation_recall import BasicSceneGraphEvaluator, evaluate_from_dict
from lib.object_detector import detector
from lib.sttran import STTran
from lib.assign_pseudo_label import prepare_func

save_result_path = args.save_result


AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, ws_object_bbox_path=None, remove_one_frame_video=True,
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=conf.num_workers, collate_fn=cuda_collate_fn,batch_size = 1)

gpu_device = torch.device('cuda')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, conf=conf).to(device=gpu_device)
object_detector.eval()

faset_rcnn_model, transforms = prepare_func()

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               transformer_mode=conf.transformer_mode,
               is_wks=conf.is_wks,
               feat_dim=conf.feat_dim,
               obj_dim=conf.obj_dim).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
use_pair_affinity = getattr(conf, 'pa_metric', False)

def make_evaluators(use_conf=None):
    """Create a set of with/semi/no evaluators. Pass conf to enable pair_affinity metric."""
    ev1 = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='with', conf=use_conf)
    ev2 = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='semi', semithreshold=0.9, conf=use_conf)
    ev3 = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=AG_dataset.object_classes,
        AG_all_predicates=AG_dataset.relationship_classes,
        AG_attention_predicates=AG_dataset.attention_relationships,
        AG_spatial_predicates=AG_dataset.spatial_relationships,
        AG_contacting_predicates=AG_dataset.contacting_relationships,
        iou_threshold=0.5,
        constraint='no', conf=use_conf)
    return ev1, ev2, ev3

# Base evaluators (without pair_affinity metric)
evaluator1, evaluator2, evaluator3 = make_evaluators(use_conf=None)

# Pair Affinity evaluators (with pair_affinity metric) - only if enabled and model has pair_affinity
if use_pair_affinity:
    eval_rel1, eval_rel2, eval_rel3 = make_evaluators(use_conf=conf)


all_frame_results = [] if save_result_path else None

with torch.no_grad():
    with tqdm(total=len(dataloader)) as t:
        for b, data in enumerate(dataloader):
            im_data = copy.deepcopy(data[0].cuda())
            im_info = copy.deepcopy(data[1].cuda())
            gt_boxes = copy.deepcopy(data[2].cuda())
            num_boxes = copy.deepcopy(data[3].cuda())
            gt_annotation = AG_dataset.gt_annotations[data[4]]
            frame_names = AG_dataset.video_list[data[4]]

            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, frame_names, faset_rcnn_model, transforms)

            if entry != None:
                pred = model(entry)
            else:
                pred = {}
            pred_dict = dict(pred)
            evaluator1.evaluate_scene_graph(gt_annotation, pred_dict, frame_names)
            evaluator2.evaluate_scene_graph(gt_annotation, pred_dict, frame_names)
            evaluator3.evaluate_scene_graph(gt_annotation, pred_dict, frame_names)

            if use_pair_affinity:
                eval_rel1.evaluate_scene_graph(gt_annotation, pred_dict, frame_names)
                eval_rel2.evaluate_scene_graph(gt_annotation, pred_dict, frame_names)
                eval_rel3.evaluate_scene_graph(gt_annotation, pred_dict, frame_names)

            # ── save per-frame results ──
            if save_result_path and pred_dict:
                # softmax/sigmoid already applied inside evaluator, do it here too
                att_dist = nn.functional.softmax(pred_dict['attention_distribution'], dim=1).cpu().numpy()
                spa_dist = pred_dict['spatial_distribution'].cpu().numpy()
                con_dist = pred_dict['contacting_distribution'].cpu().numpy()
                pair_idx = pred_dict['pair_idx'].cpu().numpy()
                im_idx = pred_dict['im_idx'].cpu().numpy()
                boxes = pred_dict['boxes'][:, 1:].cpu().numpy()
                if conf.mode == 'predcls':
                    pred_classes = pred_dict['labels'].cpu().numpy()
                    pred_scores = pred_dict['scores'].cpu().numpy()
                else:
                    pred_classes = pred_dict['pred_labels'].cpu().numpy()
                    pred_scores = pred_dict['pred_scores'].cpu().numpy()
                pa_scores = None
                if 'pair_affinity' in pred_dict:
                    pa_scores = torch.sigmoid(pred_dict['pair_affinity']).cpu().numpy().squeeze(-1)

                for idx, frame_gt in enumerate(gt_annotation):
                    frame_mask = (im_idx == idx)
                    # GT info
                    num_gt_objects = len(frame_gt) - 1  # exclude person
                    num_gt_rels = 0
                    for obj_anno in frame_gt[1:]:
                        num_gt_rels += 1  # attention (1 per object)
                        num_gt_rels += len(obj_anno['spatial_relationship'])
                        num_gt_rels += len(obj_anno['contacting_relationship'])

                    # Detected pairs for this frame
                    frame_pair_idx = pair_idx[frame_mask]
                    num_det_pairs = frame_pair_idx.shape[0]
                    # neg pairs = detected pairs - GT object count (each GT object = 1 positive pair)
                    num_neg_pairs = max(0, num_det_pairs - num_gt_objects)

                    # Per-frame box indices (remap to local)
                    frame_box_ids = np.unique(frame_pair_idx)
                    id_map = {old: new for new, old in enumerate(frame_box_ids)}
                    local_pair_idx = np.array([[id_map[p[0]], id_map[p[1]]] for p in frame_pair_idx]) if frame_pair_idx.size > 0 else np.array([])

                    frame_record = {
                        'frame_name': frame_names[idx],
                        'video_idx': int(data[4]),
                        # GT
                        'num_gt_objects': num_gt_objects,
                        'num_gt_rels': num_gt_rels,
                        'gt_annotation': frame_gt,
                        # Predictions (per-frame local)
                        'pred_boxes': boxes[frame_box_ids] if frame_box_ids.size > 0 else np.array([]),
                        'pred_classes': pred_classes[frame_box_ids] if frame_box_ids.size > 0 else np.array([]),
                        'pred_scores': pred_scores[frame_box_ids] if frame_box_ids.size > 0 else np.array([]),
                        'pair_idx': local_pair_idx,
                        'attention_distribution': att_dist[frame_mask],
                        'spatial_distribution': spa_dist[frame_mask],
                        'contacting_distribution': con_dist[frame_mask],
                        'pair_affinity': pa_scores[frame_mask] if pa_scores is not None else None,
                        # Pair stats
                        'num_det_pairs': num_det_pairs,
                        'num_neg_pairs': num_neg_pairs,
                    }
                    all_frame_results.append(frame_record)

            t.update(1)


print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------semi constraint-------------------------------')
evaluator2.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator3.print_stats()

if use_pair_affinity:
    print('')
    print('============ With Pair Affinity Metric ============')
    print('-------------------------with constraint-------------------------------')
    eval_rel1.print_stats()
    print('-------------------------semi constraint-------------------------------')
    eval_rel2.print_stats()
    print('-------------------------no constraint-------------------------------')
    eval_rel3.print_stats()

# save res
with_res = evaluator1.save_stats()
semi_res = evaluator2.save_stats()
no_res = evaluator3.save_stats()
res = {'with': with_res, 'semi': semi_res, 'no': no_res}
if use_pair_affinity:
    with_rel_res = eval_rel1.save_stats()
    semi_rel_res = eval_rel2.save_stats()
    no_rel_res = eval_rel3.save_stats()
    res['with_rel'] = with_rel_res
    res['semi_rel'] = semi_rel_res
    res['no_rel'] = no_rel_res
test_res = {}
test_res[conf.model_path] = res

# ── Save detailed per-frame results ──
if save_result_path:
    # Sort by num_neg_pairs for easy top/bottom 25% slicing
    all_frame_results.sort(key=lambda x: x['num_neg_pairs'])
    n = len(all_frame_results)
    q25 = n // 4
    save_data = {
        'config': args.cfg_file,
        'model_path': conf.model_path,
        'mean_recall': res,
        'mean_recall_pa': {
            'with': eval_rel1.save_stats() if use_pair_affinity else None,
            'semi': eval_rel2.save_stats() if use_pair_affinity else None,
            'no': eval_rel3.save_stats() if use_pair_affinity else None,
        },
        'per_frame': all_frame_results,
        'num_frames': len(all_frame_results),
        'neg_pair_stats': {
            'bottom_25_indices': list(range(q25)),
            'top_25_indices': list(range(n - q25, n)),
            'bottom_25_neg_range': (all_frame_results[0]['num_neg_pairs'],
                                     all_frame_results[q25 - 1]['num_neg_pairs']) if q25 > 0 else None,
            'top_25_neg_range': (all_frame_results[n - q25]['num_neg_pairs'],
                                  all_frame_results[-1]['num_neg_pairs']) if q25 > 0 else None,
        },
        'object_classes': AG_dataset.object_classes,
        'relationship_classes': AG_dataset.relationship_classes,
        'attention_relationships': AG_dataset.attention_relationships,
        'spatial_relationships': AG_dataset.spatial_relationships,
        'contacting_relationships': AG_dataset.contacting_relationships,
    }

    with open(save_result_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f'\nResult saved to: {save_result_path}')
    print(f'  Total frames: {n}')
    print(f'  Bottom 25% neg pairs: {save_data["neg_pair_stats"]["bottom_25_neg_range"]}')
    print(f'  Top 25% neg pairs: {save_data["neg_pair_stats"]["top_25_neg_range"]}')

