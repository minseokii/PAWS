from platform import release
import random
import torch
import os
import cv2
import numpy as np
import json
from random import choice
import sys
sys.path.append('/SSD1/minseok/WS-DSGG/TRKT/PLA/lib/')
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import config_dataset_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from lib.extract_bbox_features import extract_feature_given_bbox, extract_feature_given_bbox_video, extract_feature_given_bbox_base_feat
import pdb
import random


def load_feature(frame_names, union_box_feature, det_path, feat_path='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/AG_detection_results_refine', is_train=True, load_feat=True):
    """
    frame_names: a list of name like '001YG.mp4/000093.png'
    """
    if is_train:
        det_total_paths = [os.path.join(det_path, f) for f in frame_names]
    else:
        det_total_paths = [os.path.join(feat_path, f) for f in frame_names]
    feat_total_paths = [os.path.join(feat_path, f) for f in frame_names]
    dets_list = []
    feat_list = [] if load_feat else None
    base_feat_list = []
    for i, p in enumerate(det_total_paths):
        dets_path = os.path.join(p, 'dets.npy')
        feat_path = os.path.join(feat_total_paths[i], 'feat.npy') if load_feat else None
        dets = np.load(dets_path, allow_pickle=True)
        dets_list.append(dets)
        if load_feat:
            feat = np.load(feat_path)
            feat_list.append(feat)
            
            
    return dets_list, feat_list, None # base_feat_list


def assign_label_to_proposals_by_dict_for_image(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way):
    
    # 先遍历一遍检查人
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        # final_people_idx上一步是在people_cong_list的index，要转换一下
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        # print("cannot find people")
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]
        
    # 获取gt中label列表
    gt_ag_class_list = []
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    # 获取在gt中有对象的object列表
    object_idx = []
    object_det = []
    object_feat = []
    for bbox_idx, bbox_det in enumerate(img_det):
        # 排除人
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]
        # 区分train和test，train的时候要和gt比较才加入，test只要类别在ag中就加入
        # 考虑oi中类别对应多个ag中类别
        if is_train:
            bbox_ag_class_list = list(set(bbox_ag_class_list) & set(gt_ag_class_list))
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])
        else:
            if len(bbox_ag_class_list) > 0:
                for c in bbox_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
                    object_det.append(bbox_det.copy())
                    object_feat.append(img_feat[bbox_idx])
    return people_det, people_feat, object_det, object_feat


def assign_label_to_proposals_by_dict_for_video(dets, feats, is_train, gt_annotation, dict_path='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/weak/', pseudo_way=0, match_mode='ori', conf=None):

    cls_dict = np.load(os.path.join(dict_path, 'ag_to_oi_word_map_synset.npy'), allow_pickle=True).tolist()
    oi_to_ag_cls_dict = np.load(os.path.join(dict_path, 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    video_people_det = []
    video_people_feat = []
    video_object_det = []
    video_object_feat = []

    for i in range(len(dets)):
        # people_det, people_feat, object_det, object_feat = assign_label_to_proposals_by_dict_for_image(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way)

        # match mode selection
        if match_mode == 'gtmatch':
            people_det, people_feat, object_det, object_feat = assign_label_for_image_wkdet_gtmatch(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way)
        elif match_mode == 'gdinomatch':
            rel_th = conf.reliability_threshold if hasattr(conf, 'reliability_threshold') else 0.4
            match_th = conf.match_threshold if hasattr(conf, 'match_threshold') else 0.1

            people_det, people_feat, object_det, object_feat = assign_label_for_image_wkdet_gdinomatch(
                dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way,
                reliability_threshold=rel_th, matching_threshold=match_th
            )
        elif match_mode == 'exact':
            # Exact match: annotation bbox와 완전 동일한 detection만 선택
            bbox_tol = conf.bbox_tolerance if hasattr(conf, 'bbox_tolerance') else 1e-3
            people_det, people_feat, object_det, object_feat = assign_label_for_image_wkdet_exactmatch(
                dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way,
                bbox_tolerance=bbox_tol
            )
        else:
            people_det, people_feat, object_det, object_feat = assign_label_to_proposals_by_dict_for_image_wkdet(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict, pseudo_way)

        if getattr(conf, 'unmatched_sampling', False) and is_train and len(object_det) > 0:
            object_det, object_feat = add_all_unmatched_objects(
                dets[i], feats[i], people_det, object_det, object_feat
            )


        video_people_det.append(people_det)
        video_people_feat.append(people_feat)
        video_object_det.append(object_det)
        video_object_feat.append(object_feat)


    return video_people_det, video_people_feat, video_object_det, video_object_feat


def assign_label_to_proposals_by_dict_for_image_wkdet(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way):
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] == 1:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_feat = img_feat[final_people_idx]
    else:
        # print("cannot find people")
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]
        

    gt_ag_class_list = []
    
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    object_idx = []
    object_det = []
    object_feat = []
    for bbox_idx, bbox_det in enumerate(img_det):

        if bbox_idx == final_people_idx or bbox_det['class'] == 1:
            continue

        if is_train:
            if bbox_det['class'] in gt_ag_class_list:
                object_idx.append(bbox_idx)
                object_det.append(bbox_det.copy())
                object_feat.append(img_feat[bbox_idx])
        else:
            object_idx.append(bbox_idx)
            object_det.append(bbox_det.copy())
            object_feat.append(img_feat[bbox_idx])

    return people_det, people_feat, object_det, object_feat


def bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    box1, box2: [x, y, w, h] or [x1, y1, x2, y2]
    """
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4 and len(box2) == 4:
        # Assume detection boxes are [x1, y1, x2, y2] format
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        return 0.0

    # Intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def assign_label_for_image_wkdet_gdinomatch(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way, reliability_threshold=0.4, matching_threshold=0.1):
    """
    Reliability threshold와 matching threshold를 사용한 GDINO matching

    Args:
        reliability_threshold: reliability score 임계값 (default: 0.4)
        matching_threshold: match score 임계값 (default: 0.1)

    Logic:
        - Reliability < threshold: 모든 class match된 object 사용
        - Reliability >= threshold: match_score 가장 높은 1개만 사용 (match_score >= matching_threshold 조건)
    """
    # 1. Person 선택 (기존과 동일)
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] == 1:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)

    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_feat = img_feat[final_people_idx]
    else:
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]

    # 2. Object 선택 - reliability와 match threshold 기반
    object_idx = []
    object_det = []
    object_feat = []

    if not is_train:
        # Test 시에는 모든 object 선택
        for bbox_idx, bbox_det in enumerate(img_det):
            if bbox_idx == final_people_idx or bbox_det['class'] == 1:
                continue
            object_idx.append(bbox_idx)
            object_det.append(bbox_det.copy())
            object_feat.append(img_feat[bbox_idx])
    else:
        # Train 시: GT annotation 기반으로 matching
        # GT class 리스트 추출
        gt_class_list = []
        for pair_info in img_gt_annotation:
            if 'class' in pair_info:
                gt_class_list.append(pair_info['class'])

        # Matched detections 선택
        matched_dets = []
        used_det_ids = set()

        # GT object별로 처리
        for gt_obj_class in gt_class_list:
            # 동일 class detection 중 아직 사용되지 않은 것만
            same_class_dets = [(idx, det) for idx, det in enumerate(img_det)
                             if det.get('class', -1) == gt_obj_class
                             and idx != final_people_idx
                             and det.get('class', -1) != 1
                             and id(det) not in used_det_ids]

            if not same_class_dets:
                continue  # 매칭 가능한 detection 없음

            # Reliability 확인 (같은 class면 같은 reliability)
            reliability = same_class_dets[0][1].get('reliability', -1.0)

            if reliability < reliability_threshold:
                # Case A: 낮은 신뢰도 → 모두 추가 (class match)
                for idx, det in same_class_dets:
                    matched_dets.append((idx, det))
                    used_det_ids.add(id(det))
            else:
                # Case B: 높은 신뢰도 → best match만 선택 (GDINO match)
                best_idx, best_det = max(same_class_dets,
                                        key=lambda x: x[1].get('match_score', -1.0))

                if best_det.get('match_score', -1.0) >= matching_threshold:
                    matched_dets.append((best_idx, best_det))
                    used_det_ids.add(id(best_det))
                # else: no match (아무것도 추가 안함)

        # 결과 정리
        for idx, det in matched_dets:
            object_idx.append(idx)
            object_det.append(det.copy())
            object_feat.append(img_feat[idx])

    return people_det, people_feat, object_det, object_feat


def assign_label_for_image_wkdet_gtmatch(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way):
    """
    IoU 기반으로 GT bbox와 가장 잘 매칭되는 detection만 선택
    """
    # 1. Person 선택 (기존과 동일)
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] == 1:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)

    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_feat = img_feat[final_people_idx]
    else:
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]

    # 2. GT에서 object bbox와 class 정보 추출
    gt_objects = []  # {'class': int, 'bbox': [x1, y1, x2, y2]}
    for pair_info in img_gt_annotation:
        if 'class' in pair_info and 'bbox' in pair_info:
            bbox = pair_info['bbox']
            # bbox를 (x, y, w, h) -> (x1, y1, x2, y2) 형식으로 변환
            if len(bbox) == 4:
                x, y, w, h = bbox
                gt_bbox = [x, y, x + w, y + h]
            else:
                gt_bbox = bbox

            gt_objects.append({
                'class': pair_info['class'],
                'bbox': gt_bbox
            })

    if not is_train:
        # Test 시에는 기존 방식과 동일
        object_idx = []
        object_det = []
        object_feat = []
        for bbox_idx, bbox_det in enumerate(img_det):
            if bbox_idx == final_people_idx or bbox_det['class'] == 1:
                continue
            object_idx.append(bbox_idx)
            object_det.append(bbox_det.copy())
            object_feat.append(img_feat[bbox_idx])

        return people_det, people_feat, object_det, object_feat

    # 3. Training 시: 각 GT object에 대해 가장 IoU가 높은 detection 선택
    object_idx = []
    object_det = []
    object_feat = []

    for gt_obj in gt_objects:
        gt_class = gt_obj['class']
        gt_bbox = gt_obj['bbox']

        # 같은 class를 가진 detection 중에서 찾기
        best_iou = 0.0
        best_idx = -1

        for bbox_idx, bbox_det in enumerate(img_det):
            # Person 제외
            if bbox_idx == final_people_idx or bbox_det['class'] == 1:
                continue

            # Class가 일치하는지 확인
            if bbox_det['class'] != gt_class:
                continue

            # IoU 계산
            det_bbox = bbox_det['rect']
            iou = bbox_iou(det_bbox, gt_bbox)

            if iou > best_iou:
                best_iou = iou
                best_idx = bbox_idx

        # 가장 IoU가 높은 detection을 pseudo label로 추가
        if best_idx != -1:
            object_idx.append(best_idx)
            object_det.append(img_det[best_idx].copy())
            object_feat.append(img_feat[best_idx])

    return people_det, people_feat, object_det, object_feat


def assign_label_for_image_wkdet_exactmatch(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict, pseudo_way, bbox_tolerance=1e-3):
    """
    Exact match: annotation bbox와 완전 동일한 (class + bbox 좌표) detection만 선택

    gdino match의 철학을 유지하면서, gdino detection이 없는 경우에도 사용 가능.
    annotation에서 나온 object bbox와 완전 동일한 detection만 pos entry로 매칭.

    Args:
        img_det: detection 결과 list
        img_feat: detection feature list
        is_train: training 여부
        img_gt_annotation: GT annotation (object_info list)
        cls_dict, oi_to_ag_cls_dict: class mapping dicts
        pseudo_way: pseudo label 방식
        bbox_tolerance: bbox 좌표 비교 시 허용 오차 (default: 1e-3)
    """
    # 1. Person 선택 (기존과 동일)
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] == 1:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)

    if len(people_conf_list) != 0:
        final_people_idx = people_conf_list.index(max(people_conf_list))
        final_people_idx = people_idx[final_people_idx]
        people_det = img_det[final_people_idx]
        people_feat = img_feat[final_people_idx]
    else:
        if pseudo_way == 0:
            return [], [], [], []
        elif pseudo_way == 1:
            final_people_idx = 0
            people_det = img_det[final_people_idx]
            people_det['class'] = 1
            people_feat = img_feat[final_people_idx]

    if not is_train:
        # Test 시에는 모든 object 선택
        object_idx = []
        object_det = []
        object_feat = []
        for bbox_idx, bbox_det in enumerate(img_det):
            if bbox_idx == final_people_idx or bbox_det['class'] == 1:
                continue
            object_idx.append(bbox_idx)
            object_det.append(bbox_det.copy())
            object_feat.append(img_feat[bbox_idx])

        return people_det, people_feat, object_det, object_feat

    # 2. GT에서 object bbox와 class 정보 추출
    gt_objects = []  # {'class': int, 'bbox': [x1, y1, x2, y2]}
    for pair_info in img_gt_annotation:
        if 'class' in pair_info and 'bbox' in pair_info:
            bbox = pair_info['bbox']
            # bbox 형식 확인 및 변환
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                # 이미 [x1, y1, x2, y2] 형식이라고 가정
                # annotation의 bbox는 보통 [x1, y1, x2, y2] 형식
                gt_bbox = list(bbox)
            else:
                gt_bbox = list(bbox) if hasattr(bbox, '__iter__') else bbox

            gt_objects.append({
                'class': pair_info['class'],
                'bbox': gt_bbox
            })

    # 3. Training 시: exact match - class와 bbox가 완전히 동일한 detection만 선택
    object_idx = []
    object_det = []
    object_feat = []
    used_det_indices = set()

    for gt_obj in gt_objects:
        gt_class = gt_obj['class']
        gt_bbox = gt_obj['bbox']

        # 같은 class를 가진 detection 중에서 exact bbox match 찾기
        for bbox_idx, bbox_det in enumerate(img_det):
            # 이미 사용된 detection 제외
            if bbox_idx in used_det_indices:
                continue

            # Person 제외
            if bbox_idx == final_people_idx or bbox_det['class'] == 1:
                continue

            # Class가 일치하는지 확인
            if bbox_det['class'] != gt_class:
                continue

            # Exact bbox match 확인
            det_bbox = bbox_det['rect']

            # bbox 좌표 비교 (tolerance 이내면 동일하다고 판단)
            bbox_match = True
            for i in range(4):
                if abs(det_bbox[i] - gt_bbox[i]) > bbox_tolerance:
                    bbox_match = False
                    break

            if bbox_match:
                # Exact match 발견
                object_idx.append(bbox_idx)
                object_det.append(bbox_det.copy())
                object_feat.append(img_feat[bbox_idx])
                used_det_indices.add(bbox_idx)
                break  # 이 GT object에 대한 매칭 완료

    return people_det, people_feat, object_det, object_feat


def add_all_unmatched_objects(all_dets, all_feats, people_det, object_det, object_feat):
    """
    Add all unmatched objects as negative samples to the train batch.

    Args:
        all_dets: all detections for the frame [list of dicts]
        all_feats: all features for the frame [list of tensors/arrays]
        people_det: matched person detections [list of dicts]
        object_det: matched object detections [list of dicts]
        object_feat: matched object features [list of tensors/arrays]

    Returns:
        updated object_det, object_feat with all unmatched objects appended
    """

    # Find indices of already matched objects and people
    matched_indices = set()

    # Find matched object indices by comparing bbox coordinates
    for matched_obj in object_det:
        matched_bbox = matched_obj.get('bbox')
        for idx, det in enumerate(all_dets):
            det_bbox = det.get('bbox')
            if matched_bbox is not None and det_bbox is not None:
                if isinstance(matched_bbox, np.ndarray) and isinstance(det_bbox, np.ndarray):
                    if np.array_equal(matched_bbox, det_bbox):
                        matched_indices.add(idx)
                        break
                elif matched_bbox == det_bbox or (hasattr(matched_bbox, '__iter__') and hasattr(det_bbox, '__iter__') and list(matched_bbox) == list(det_bbox)):
                    matched_indices.add(idx)
                    break

    # Find person index (people_det is a single dict, not a list)
    if people_det:
        person_bbox = people_det.get('bbox')
        for idx, det in enumerate(all_dets):
            det_bbox = det.get('bbox')
            if person_bbox is not None and det_bbox is not None:
                if isinstance(person_bbox, np.ndarray) and isinstance(det_bbox, np.ndarray):
                    if np.array_equal(person_bbox, det_bbox):
                        matched_indices.add(idx)
                        break
                elif person_bbox == det_bbox or (hasattr(person_bbox, '__iter__') and hasattr(det_bbox, '__iter__') and list(person_bbox) == list(det_bbox)):
                    matched_indices.add(idx)
                    break

    # Add all unmatched objects (not matched, not person class)
    for idx, det in enumerate(all_dets):
        if idx not in matched_indices and det.get('class', -1) != 1:
            det_copy = det.copy()
            det_copy['is_negative_sample'] = True
            object_det.append(det_copy)
            object_feat.append(all_feats[idx])

    return object_det, object_feat


def assign_label_to_general_detector(dets_list):
    oi_to_ag_cls_dict = np.load(os.path.join('/network_space/server126/shared/xuzhu/CSA/clean_version/data/action-genome/annotations/weak', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()
    final_list = []
    for img_det in dets_list:
        mapped_cat = oi_to_ag_cls_dict[img_det['class']]
        if len(mapped_cat) == 0:
            continue
        for mapped_cat_id in mapped_cat:
            img_det['class'] = mapped_cat_id
            final_list.append(img_det)
    return final_list

def create_dis(conf, idx):
    distrubution = torch.zeros(36)
    distrubution[idx] = conf
    distrubution[torch.where(distrubution==0)] = (1-conf) / 35
    return distrubution


def create_dis_list(FINAL_SCORES_OI, PRED_LABELS_OI):
    oi_to_ag_cls_dict = np.load(os.path.join('/network_space/server126/shared/xuzhu/CSA/clean_version/data/action-genome/annotations/weak', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    all_ag_id = list(range(2, 36))
    dis_ag = torch.zeros((len(FINAL_SCORES_OI), 36), device=FINAL_SCORES_OI.device)
    for i in range(len(FINAL_SCORES_OI)):
        conf = FINAL_SCORES_OI[i].item()
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[PRED_LABELS_OI[i].item()]
        if bbox_ag_class_list != []:
            idx = random.choice(bbox_ag_class_list)
        else:
            idx = random.choice(all_ag_id)
            # 直接取概率最高的table
            # idx = object_freq[0]
        dis_ag[i] = create_dis(conf, idx-1)

    return dis_ag


def category_oi2ag(dis_oi):
    oi_to_ag_cls_dict = np.load(os.path.join('/network_space/server126/shared/xuzhu/CSA/clean_version/data/action-genome/annotations/weak', 'oi_to_ag_word_map_synset.npy'), allow_pickle=True).tolist()

    dis_ag = torch.zeros((len(dis_oi), 36), device=dis_oi.device)
    for dis_id, one_dis in enumerate(dis_oi):
        for oi_id, mapped_ag_id_list in oi_to_ag_cls_dict.items():
            for ag_id in mapped_ag_id_list:
                dis_ag[dis_id][ag_id-1] += one_dis[oi_id]

    return dis_ag


def prepare_func(thresh=0.2):
    config_file = "configs/detector/vinvl_x152c4.yaml"
    opts = ["MODEL.WEIGHT", "models/vinvl/vinvl_vg_x152c4.pth", 
            "MODEL.ROI_HEADS.NMS_FILTER", "1",
            "MODEL.ROI_HEADS.SCORE_THRESH", str(thresh),
            "DATA_DIR", "datasets",
            "TEST.IGNORE_BOX_REGRESSION", "False"]

    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(sg_cfg)
    cfg.set_new_allowed(False)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)

    model = AttrRCNN(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    checkpointer.load(cfg.MODEL.WEIGHT)

    transforms = build_transforms(cfg, is_train=False)

    return model, transforms


def convert_data(is_train, base_feat_list, video_people_det, video_people_feat, video_object_det, video_object_feat, \
    gt_annotation, frame_names, faset_rcnn_model, transforms, union_box_feature, conf):
    # 将video_people_det, video_people_feat, video_object_det, video_object_feat转换成entry的格式

    frame_num = len(video_people_det)
    bbox_num = 0

    feat_dim = conf.obj_dim if hasattr(conf, 'obj_dim') else 192

    for idx in range(frame_num):
        if video_people_det[idx] != []:
            bbox_num += 1
            bbox_num += len(video_object_det[idx])

    # bbox_num = 0
    MyDevice = torch.device('cuda:0')
    boxes = torch.zeros((bbox_num, 5), device=MyDevice)

    labels = torch.zeros(bbox_num, dtype=torch.int64, device=MyDevice)
    # obj_labels = torch.zeros(bbox_num-frame_num, dtype=torch.int64, device=MyDevice)
    scores = torch.zeros(bbox_num, device=MyDevice)
    distribution = torch.zeros((bbox_num, 36), device=MyDevice)
    features = torch.zeros((bbox_num, feat_dim), device=MyDevice) # TODO: ours 192, baseline or roi feature 2048
    im_idx = []
    pair_idx = []
    a_rel = []
    s_rel = []
    c_rel = []
    rel_gt = []
    negative_mask = []  # True = exclude from loss, False = include in loss
    box_idx = []

    bbox_cnt = 0
    for idx in range(frame_num):

        if video_people_det[idx] != []:
            people_det = video_people_det[idx]
            people_feat = video_people_feat[idx]
            object_det = video_object_det[idx]
            object_feat = video_object_feat[idx]
            
            # 构造 boxes labels scores distrubution features
            boxes[bbox_cnt][0] = idx
            boxes[bbox_cnt][1:5] = torch.Tensor(people_det['rect'])
            labels[bbox_cnt] = people_det['class']
            scores[bbox_cnt] = people_det['conf']
            distribution[bbox_cnt] = create_dis(people_det['conf'], people_det['class'] - 1)  # because '__background__' is not a label
            if type(people_feat) != torch.Tensor:
                features[bbox_cnt] = torch.from_numpy(people_feat)
            else:
                features[bbox_cnt] = people_feat

            people_bbox_idx = bbox_cnt # 记录people的序号，之后im_idx要用
            box_idx.append(idx)
            bbox_cnt += 1

            for bbox_det, bbox_feat in zip(object_det, object_feat):
                boxes[bbox_cnt][0] = idx
                boxes[bbox_cnt][1:5] = torch.Tensor(bbox_det['rect'])
                labels[bbox_cnt] = bbox_det['class']
                scores[bbox_cnt] = bbox_det['conf']
                distribution[bbox_cnt] = create_dis(bbox_det['conf'], bbox_det['class'] - 1)  # because '__background__' is not a label
                if type(bbox_feat) != torch.Tensor:
                    features[bbox_cnt] = torch.from_numpy(bbox_feat)
                else:
                    features[bbox_cnt] = bbox_feat
            
                # 构造 im_idx pair_idx
                '''
                img_gt_annotation = gt_annotation[idx]
                for obj_info in img_gt_annotation:
                    if 'class' in obj_info:
                        if obj_info['class'] == bbox_det['class']:
                            # 在gt中找到对应的object
                            im_idx.append(idx)
                            pair_idx.append([people_bbox_idx, bbox_cnt])
                            a_rel.append(obj_info['attention_relationship'].tolist())
                            s_rel.append(obj_info['spatial_relationship'].tolist())
                            c_rel.append(obj_info['contacting_relationship'].tolist())
                '''
                img_gt_annotation = gt_annotation[idx]
                # 注意warning：这里im_idx和pair_idx，只有training时候才筛选，testing的时候不筛选
                # testing的时候，也不需要pseudo gt了
                if is_train:
                    # Check if this is a negative sample
                    if bbox_det.get('is_negative_sample', False):
                        # Negative sample: create pair but exclude from loss
                        im_idx.append(idx)
                        pair_idx.append([people_bbox_idx, bbox_cnt])
                        # Dummy labels (won't be used)
                        a_rel.append([0])  # 3 attention classes
                        s_rel.append([0])  # 6 spatial classes
                        c_rel.append([0]) # 17 contacting classes
                        rel_gt.append(False)      # Use fusion label (not hard GT)
                        negative_mask.append(True)  # Exclude from loss calculation
                    else:
                        # Normal matched object: find GT annotation
                        for obj_info in img_gt_annotation:
                            if 'class' in obj_info:
                                if obj_info['class'] == bbox_det['class']:
                                    # 在gt中找到对应的object
                                    im_idx.append(idx)
                                    pair_idx.append([people_bbox_idx, bbox_cnt])
                                    a_rel.append(obj_info['attention_relationship'].tolist())
                                    s_rel.append(obj_info['spatial_relationship'].tolist())
                                    c_rel.append(obj_info['contacting_relationship'].tolist())
                                    if obj_info['object_source']['ar'][-1] == '1gt' and obj_info['object_source']['sr'][-1] == '1gt' and obj_info['object_source']['cr'][-1] == '1gt':
                                        rel_gt.append(True)
                                    elif obj_info['object_source']['ar'][-1] == 'gt' and obj_info['object_source']['sr'][-1] == 'gt' and obj_info['object_source']['cr'][-1] == 'gt':
                                        rel_gt.append(True)
                                    elif obj_info['object_source']['ar'][-1] == '1' and obj_info['object_source']['sr'][-1] == '1' and obj_info['object_source']['cr'][-1] == '1':
                                        rel_gt.append(False)
                                    else:
                                        print(obj_info['object_source']['ar'][-1], obj_info['object_source']['sr'][-1], obj_info['object_source']['cr'][-1], 'Error!')
                                    negative_mask.append(False)  # Normal object: include in loss
                                    break
                else:
                    im_idx.append(idx)
                    pair_idx.append([people_bbox_idx, bbox_cnt])

                box_idx.append(idx)
                bbox_cnt += 1

    rel_gt = torch.tensor(rel_gt, device=MyDevice)
    negative_mask = torch.tensor(negative_mask, device=MyDevice)
    box_idx = torch.tensor(box_idx, device=MyDevice)
    im_idx = torch.tensor(im_idx, device=MyDevice)
    pair_idx = torch.tensor(pair_idx, device=MyDevice).long()

    rel_num = len(pair_idx)
    if rel_num == 0:
        return None
    '''
    else:
        return {'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': torch.zeros((rel_num, 2048, 7, 7), device=MyDevice),
            'spatial_masks': torch.zeros((rel_num, 2, 27, 27), device=MyDevice),
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel}
    '''
    if union_box_feature:
        # for detection union boxes
        imgs_paths = [os.path.join('/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/frames/', f) for f in frame_names]
        cv2_imgs = [cv2.imread(img_file) for img_file in imgs_paths]
        union_boxes = torch.cat((im_idx[:, None],
                                torch.min(boxes[:, 1:3][pair_idx[:, 0]],
                                        boxes[:, 1:3][pair_idx[:, 1]]),
                                torch.max(boxes[:, 3:5][pair_idx[:, 0]],
                                        boxes[:, 3:5][pair_idx[:, 1]])), 1)
        # Lowres consistency: if conf.lowres_factor > 1, downsample images and bboxes
        # so union_feat is extracted at the same resolution as the cached per-object feat.
        lowres_factor = getattr(conf, 'lowres_factor', 1) if conf is not None else 1
        if lowres_factor and lowres_factor > 1:
            cv2_imgs = [cv2.resize(im, (max(8, im.shape[1] // lowres_factor), max(8, im.shape[0] // lowres_factor)),
                                   interpolation=cv2.INTER_AREA) for im in cv2_imgs]
            union_boxes = union_boxes.clone()
            union_boxes[:, 1:] = union_boxes[:, 1:] / float(lowres_factor)
        union_boxes_list = [union_boxes[union_boxes[:, 0] == i] for i in range(frame_num)]
        union_feat_list = []

        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                union_feat_list.append(extract_feature_given_bbox(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:]))
                # union_feat_list.append(extract_feature_given_bbox_base_feat(faset_rcnn_model, transforms, cv2_imgs[i], union_boxes_list[i][:, 1:], base_feat_list[i]))
            else:
                union_feat_list.append(torch.Tensor([]).cuda(0))
        union_feat = torch.cat(union_feat_list)
        '''
        imgs = []
        bboxes = []
        for i, union_boxes_one_image in enumerate(union_boxes_list):
            if len(union_boxes_list[i]) > 0:
                imgs.append(cv2_imgs[i])
                bboxes.append(union_boxes_list[i][:, 1:])
        # bboxes = union_boxes_list[:][:, 1:]
        union_feat_list = extract_feature_given_bbox_video(faset_rcnn_model, transforms, cv2_imgs, bboxes)
        union_feat = union_feat_list
        '''

    else:
        union_feat = torch.zeros((rel_num, 2048, 7, 7), device=MyDevice)
        # union_feat = torch.randn(rel_num, 2048, 7, 7).cuda(0)

    if pair_idx.shape[0] == 0:
        spatial_masks = torch.zeros((rel_num, 2, 27, 27), device=MyDevice)
    else:
        pair_rois = torch.cat((boxes[pair_idx[:,0],1:],boxes[pair_idx[:,1],1:]), 1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5, device=MyDevice)
            
    obj_labels = labels[labels != 1]
    # obj_boxes = boxes[labels != 1]
    
    entry = {'boxes': boxes,
            'labels': labels,
            'obj_labels': obj_labels,
            'scores': scores,
            'distribution': distribution,
            'im_idx': im_idx,
            'pair_idx': pair_idx,
            'features': features,
            'union_feat': union_feat,
            'spatial_masks': spatial_masks,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel,
            'rel_gt': rel_gt,
            'negative_mask': negative_mask,
            'box_idx': box_idx}

    return entry


#############################################
# test the detector
#############################################

def entry_to_pred(entry):
    # convert entry to pred directly
    if entry == None:
        return {}

    entry['pred_labels'] = entry['labels']
    entry['pred_scores'] = entry['scores']
    rel_num = len(entry['attention_gt'])
    attention_distribution = torch.zeros(rel_num, 3).cuda(0)
    spatial_distribution = torch.zeros(rel_num, 6).cuda(0)
    contacting_distribution = torch.zeros(rel_num, 17).cuda(0)

    for i in range(rel_num):
        # attention_distribution[i][entry['attention_gt'][i]] = 1 / len(entry['attention_gt'][i])
        # spatial_distribution[i][entry['spatial_gt'][i]] = 1 / len(entry['spatial_gt'][i])
        # contacting_distribution[i][entry['contacting_gt'][i]] = 1 / len(entry['contacting_gt'][i])
        attention_distribution[i][entry['attention_gt'][i]] = 1
        spatial_distribution[i][entry['spatial_gt'][i]] = 1
        contacting_distribution[i][entry['contacting_gt'][i]] = 1

    entry['attention_distribution'] = attention_distribution
    entry['spatial_distribution'] = spatial_distribution
    entry['contacting_distribution'] = contacting_distribution

    return entry



#############################################
# debug
#############################################

def count_person_and_object_for_image(img_det, img_feat, is_train, img_gt_annotation, cls_dict, oi_to_ag_cls_dict):
    """
    only use a dictionary to assign gt object labels
    TODO: using box location to match gt objects
    dict中有映射、gt中有对象保留，其他舍去（gt中同一个对象应该不会有两个）
    注意先检查人
    """
    
    has_person_img = True

    # 检查人的部分不需要区分训练和测试
    # 先遍历一遍检查人
    # 因为肯定有人所以不和gt比
    people_oi_idx = cls_dict[1]
    people_conf_list = []
    people_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        if bbox_det['class'] in people_oi_idx:
            people_conf_list.append(bbox_det['conf'])
            people_idx.append(bbox_idx)
    if len(people_conf_list) != 0:
        has_person_img = True
        final_people_idx = people_conf_list.index(max(people_conf_list))
        people_det = img_det[final_people_idx]
        people_det['class'] = 1
        people_feat = img_feat[final_people_idx]
    else:
        has_person_img = False
        return has_person_img, 0
        # final_people_idx = 0
        # people_det = img_det[final_people_idx]
        # people_det['class'] = 1
        # people_feat = img_feat[final_people_idx]
        
    # 获取gt中label列表
    gt_ag_class_list = []
    for pair_info in img_gt_annotation:
        if 'class' in pair_info:
            gt_ag_class_list.append(pair_info['class'])
    # 获取在gt中有对象的object列表
    object_idx = []
    for bbox_idx, bbox_det in enumerate(img_det):
        # 排除人
        if bbox_idx == final_people_idx:
            continue
        if bbox_det['class'] in people_oi_idx:
            continue
        # 获取bbox对应的ag中类别
        bbox_ag_class_list = oi_to_ag_cls_dict[bbox_det['class']]
        # 区分train和test，train的时候要和gt比较才加入，test只要类别在ag中就加入
        if is_train:
            for c in bbox_ag_class_list:
                if c in gt_ag_class_list:
                    bbox_det['class'] = c
                    object_idx.append(bbox_idx)
        else:
            if len(bbox_ag_class_list) > 0:
                c = choice(bbox_ag_class_list)
                bbox_det['class'] = c
                object_idx.append(bbox_idx)

    return has_person_img, len(object_idx)



def count_person_and_object_for_video(dets, feats, is_train, gt_annotation, cls_dict, oi_to_ag_cls_dict, frame_names):

    f_names = [f.split('/')[1] for f in frame_names]
    info_dict = {}
    no_person_img_cnt = 0
    with_person_img_cnt = 0
    total_rel_cnt = 0

    for i in range(len(dets)):
        has_person_img, rel_cnt = count_person_and_object_for_image(dets[i], feats[i], is_train, gt_annotation[i], cls_dict, oi_to_ag_cls_dict)
        info_dict[f_names[i]] = (has_person_img, rel_cnt)
        if has_person_img:
            with_person_img_cnt += 1
        else:
            no_person_img_cnt += 1
        total_rel_cnt += rel_cnt

    return info_dict, no_person_img_cnt, with_person_img_cnt, total_rel_cnt
