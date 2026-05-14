"""
Model-free pseudo label generation with:
1. AG Dataset을 사용하여 비디오 순회
2. GDino matching된 entry를 PKL annotation 형식으로 변환
3. Center frame에서 양방향으로 IOU 기반 propagation
"""

import os
import sys
sys.path.insert(0, '/SSD1/minseok/WS-DSGG/TRKT/PLA')

import copy
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from lib.config import conf, cfg_from_file
from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.assign_pseudo_label import prepare_func
from lib.ults.iou import bb_intersection_over_union


def generate_video_list(frame_name_list):
    """
    프레임 이름 리스트를 비디오별로 그룹화하고 정렬
    """
    video_dict = {}
    for frame_name in frame_name_list:
        video_name = frame_name[:5]  # 예: '001YG'
        if video_name in video_dict.keys():
            video_dict[video_name].append(frame_name)
        else:
            video_dict[video_name] = [frame_name]

    # 각 비디오 내 프레임을 정렬 (propagation 순서 보장)
    for video_name in video_dict:
        video_dict[video_name] = sorted(video_dict[video_name])

    return video_dict


def load_detections_from_path(frame_list, det_path, object_classes):
    """
    det_path에서 모든 프레임의 detection을 로드

    Args:
        frame_list: 프레임 이름 리스트
        det_path: detection 결과 경로
        object_classes: object class 이름 리스트

    Returns:
        frame_detections: {frame_idx: [{'class': ..., 'bbox': ..., 'score': ...}, ...]}
    """
    frame_detections = {i: [] for i in range(len(frame_list))}

    for frame_idx, frame_name in enumerate(frame_list):
        dets_path = os.path.join(det_path, frame_name, 'dets.npy')

        if not os.path.exists(dets_path):
            continue

        try:
            dets = np.load(dets_path, allow_pickle=True)

            # dets 구조: list of dict, 각 dict는 {'rect': [x1,y1,x2,y2], 'class': idx, 'conf': score}
            for det in dets:
                # person (class=1)은 제외
                if det['class'] == 1:
                    continue

                obj_info = {
                    'class': det['class'],
                    'class_name': object_classes[det['class']],
                    'bbox': det['rect'] if isinstance(det['rect'], list) else det['rect'].tolist(),
                    'score': det['conf'],
                }
                frame_detections[frame_idx].append(obj_info)

        except Exception as e:
            print(f'Error loading {dets_path}: {e}')
            continue

    return frame_detections


def get_label_list(frame_num, label_num):
    """
    Get label (GT) and unlabel frame indices
    Returns: label_list (GT frames), unlabel_list (frames to propagate to)
    """
    total_list = list(range(frame_num))
    if label_num >= frame_num:
        return np.array(total_list), np.array([])

    if label_num * 2 > frame_num:
        unlabel_list, label_list = get_label_list(frame_num, frame_num - label_num)
        return label_list, unlabel_list

    part_num = label_num + 1
    part_list = [i / part_num for i in range(1, part_num)]
    label_list = [round(i * (frame_num - 1)) for i in part_list]
    unlabel_list = [i for i in total_list if i not in label_list]

    return np.array(label_list), np.array(unlabel_list)


def entry_to_frame_detections(entry, frame_num, object_classes, entry_idx_to_actual_idx=None):
    """
    Entry를 프레임별 detection dict로 변환

    Entry format:
        boxes: [N, 5] where [:, 0] is frame_idx (relative to entry's frame_names), [:, 1:5] is bbox
        labels: [N] class indices
        scores: [N] confidence scores
        pair_idx: [M, 2] person-object pairs
        attention_gt, spatial_gt, contacting_gt: relation labels

    Args:
        entry: object_detector output
        frame_num: total number of frames in full video
        object_classes: list of object class names
        entry_idx_to_actual_idx: dict mapping entry's frame_idx to actual frame_list index
                                 e.g., {0: 44} means entry's frame 0 is actually frame 44 in full video

    Returns:
        frame_detections: {frame_idx: [{'class': ..., 'bbox': ..., 'relations': ...}, ...]}
    """
    frame_detections = {i: [] for i in range(frame_num)}

    if entry is None:
        return frame_detections

    boxes = entry['boxes']  # [N, 5]
    labels = entry['labels']  # [N]
    scores = entry.get('scores', torch.ones(len(labels), device=boxes.device))
    pair_idx = entry['pair_idx']  # [M, 2]
    attention_gt = entry['attention_gt']  # list of lists
    spatial_gt = entry['spatial_gt']
    contacting_gt = entry['contacting_gt']

    # pair_idx를 통해 각 object의 relation 정보를 매핑
    # pair_idx[i] = [person_idx, object_idx]
    object_to_relation = {}
    for pair_i, (person_idx, obj_idx) in enumerate(pair_idx.cpu().numpy()):
        obj_idx = int(obj_idx)
        object_to_relation[obj_idx] = {
            'attention_relationship': np.array(attention_gt[pair_i]),
            'spatial_relationship': np.array(spatial_gt[pair_i]),
            'contacting_relationship': np.array(contacting_gt[pair_i]),
        }

    # 각 bbox를 프레임별로 분류
    for bbox_idx in range(len(boxes)):
        entry_frame_idx = int(boxes[bbox_idx, 0].item())  # entry 내에서의 frame index

        # entry의 frame_idx를 실제 frame_list 내 위치로 매핑
        if entry_idx_to_actual_idx is not None:
            actual_frame_idx = entry_idx_to_actual_idx.get(entry_frame_idx, None)
            if actual_frame_idx is None:
                continue  # 매핑 실패 시 스킵
        else:
            actual_frame_idx = entry_frame_idx

        bbox = boxes[bbox_idx, 1:5].cpu().numpy()
        label = int(labels[bbox_idx].item())
        score = float(scores[bbox_idx].item())

        # person (label=1)은 제외, object만 처리
        if label == 1:  # person class
            continue

        # relation 정보가 있는 object만 추가 (GDino matched)
        if bbox_idx in object_to_relation:
            obj_info = {
                'class': label,  # class index
                'class_name': object_classes[label],
                'bbox': bbox.tolist(),
                'score': score,
                'attention_relationship': object_to_relation[bbox_idx]['attention_relationship'],
                'spatial_relationship': object_to_relation[bbox_idx]['spatial_relationship'],
                'contacting_relationship': object_to_relation[bbox_idx]['contacting_relationship'],
                'object_source': {
                    'bbox': ['de'],
                    'ar': ['1gt'],
                    'sr': ['1gt'],
                    'cr': ['1gt'],
                }
            }
            frame_detections[actual_frame_idx].append(obj_info)

    return frame_detections


def propagate_relations(all_frame_detections, center_objects, center_idx, frame_num, iou_threshold=0.5, allow_skip=True):
    """
    Center frame에서 양방향으로 relation propagation

    Args:
        all_frame_detections: {frame_idx: [obj_info, ...]} - 모든 프레임의 detection (det_path에서 로드)
        center_objects: center frame의 objects with relations (entry에서 추출)
        center_idx: center frame index
        frame_num: total number of frames
        iou_threshold: IOU threshold for matching
        allow_skip: if True, continue propagation even when matching fails
                   if False, stop propagation when matching fails and move to next direction

    Returns:
        propagated_detections: {frame_idx: [obj_info, ...]}
    """
    propagated = {i: [] for i in range(frame_num)}

    # Center frame: relation이 있는 objects 저장
    propagated[center_idx] = copy.deepcopy(center_objects)

    stats = {'success': 0, 'skip': 0, 'early_stop': 0}

    # Forward propagation (center -> 0)
    front_list = list(range(center_idx - 1, -1, -1))
    prev_objects = propagated[center_idx]

    for frame_idx in front_list:
        current_detections = all_frame_detections[frame_idx]  # det_path에서 로드한 detection
        matched_objects = []

        for curr_det in current_detections:
            curr_bbox = curr_det['bbox']
            curr_class = curr_det['class']

            # 이전 프레임의 같은 class object와 IOU 매칭
            for prev_obj in prev_objects:
                if curr_class == prev_obj['class']:
                    iou = bb_intersection_over_union(curr_bbox, prev_obj['bbox'])
                    if iou > iou_threshold:
                        # 현재 detection에 이전 object의 relation 복사
                        matched_obj = copy.deepcopy(curr_det)
                        matched_obj['attention_relationship'] = copy.deepcopy(prev_obj['attention_relationship'])
                        matched_obj['spatial_relationship'] = copy.deepcopy(prev_obj['spatial_relationship'])
                        matched_obj['contacting_relationship'] = copy.deepcopy(prev_obj['contacting_relationship'])
                        matched_obj['object_source'] = {
                            'bbox': ['de'],
                            'ar': ['1'],
                            'sr': ['1'],
                            'cr': ['1'],
                        }
                        matched_objects.append(matched_obj)
                        break

        if len(matched_objects) > 0:
            propagated[frame_idx] = matched_objects
            prev_objects = matched_objects
            stats['success'] += 1
        else:
            if allow_skip:
                # Skip but keep prev_objects for next iteration
                stats['skip'] += 1
            else:
                # Stop forward propagation and move to backward
                stats['early_stop'] += 1
                break

    # Backward propagation (center -> end)
    behind_list = list(range(center_idx + 1, frame_num))
    prev_objects = propagated[center_idx]

    for frame_idx in behind_list:
        current_detections = all_frame_detections[frame_idx]
        matched_objects = []

        for curr_det in current_detections:
            curr_bbox = curr_det['bbox']
            curr_class = curr_det['class']

            for prev_obj in prev_objects:
                if curr_class == prev_obj['class']:
                    iou = bb_intersection_over_union(curr_bbox, prev_obj['bbox'])
                    if iou > iou_threshold:
                        matched_obj = copy.deepcopy(curr_det)
                        matched_obj['attention_relationship'] = copy.deepcopy(prev_obj['attention_relationship'])
                        matched_obj['spatial_relationship'] = copy.deepcopy(prev_obj['spatial_relationship'])
                        matched_obj['contacting_relationship'] = copy.deepcopy(prev_obj['contacting_relationship'])
                        matched_obj['object_source'] = {
                            'bbox': ['de'],
                            'ar': ['1'],
                            'sr': ['1'],
                            'cr': ['1'],
                        }
                        matched_objects.append(matched_obj)
                        break

        if len(matched_objects) > 0:
            propagated[frame_idx] = matched_objects
            prev_objects = matched_objects
            stats['success'] += 1
        else:
            if allow_skip:
                # Skip but keep prev_objects for next iteration
                stats['skip'] += 1
            else:
                # Stop backward propagation
                stats['early_stop'] += 1
                break

    return propagated, stats


def to_python_native(obj):
    """
    numpy array 및 numpy scalar를 Python 기본 타입으로 재귀적으로 변환
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_python_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_native(item) for item in obj)
    else:
        return obj


def convert_to_pkl_format(propagated_detections, frame_list, weak_anno, obj_classes,
                          attention_rels, spatial_rels, contacting_rels):
    """
    Propagated detections를 PKL annotation 형식으로 변환

    Args:
        propagated_detections: {frame_idx: [obj_info, ...]}
        frame_list: 전체 프레임 이름 리스트
        weak_anno: 원본 weak annotation (person_info 가져오기 위함)
        obj_classes: object class names
        attention_rels: attention relationship names (index -> str)
        spatial_rels: spatial relationship names
        contacting_rels: contacting relationship names

    PKL format:
    {
        'frame_name': {
            'person_info': {'bbox': [x1, y1, x2, y2]},
            'object_info': [...]
        },
        ...
    }
    """
    result = {}

    for frame_idx, frame_name in enumerate(frame_list):
        # Person bbox from weak_anno (모든 프레임에 대해)
        frame_weak_anno = weak_anno.get(frame_name, {})
        person_info = frame_weak_anno.get('person_info', {})
        person_bbox = person_info.get('bbox', [[0, 0, 100, 100]])
        bbox_size = person_info.get('bbox_size', (1920, 1080))  # 원본에서 bbox_size 가져오기

        # numpy array를 list로 변환
        person_bbox = to_python_native(person_bbox)
        bbox_size = to_python_native(bbox_size)

        # Object info (propagated된 경우만 있음)
        object_info = []
        for obj in propagated_detections.get(frame_idx, []):
            # Relationship indices를 문자열로 변환
            att_rel = to_python_native(obj['attention_relationship'])
            spa_rel = to_python_native(obj['spatial_relationship'])
            con_rel = to_python_native(obj['contacting_relationship'])

            att_rel_str = [attention_rels[i] for i in att_rel]
            spa_rel_str = [spatial_rels[i] for i in spa_rel]
            con_rel_str = [contacting_rels[i] for i in con_rel]

            # bbox도 변환
            bbox = to_python_native(obj['bbox'])

            obj_entry = {
                'class': obj_classes[obj['class']],
                'bbox': bbox,
                'attention_relationship': att_rel_str,
                'spatial_relationship': spa_rel_str,
                'contacting_relationship': con_rel_str,
                'visible': True,
                'metadata': {'set': 'train'},  # train set으로 지정
                'object_source': obj.get('object_source', {
                    'bbox': ['de'],
                    'ar': ['1'],
                    'sr': ['1'],
                    'cr': ['1'],
                })
            }
            object_info.append(obj_entry)

        result[frame_name] = {
            'person_info': {'bbox': person_bbox, 'bbox_size': bbox_size},
            'object_info': object_info
        }

    return result


def generate_pseudo_labels(
    config_path='configs/config_3.yml',
    output_path='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/weak/gt_annotation_gdino_modelfree_05_skip.pkl',
    iou_threshold=0.5,
    gpu_id=0,
    label_num=1,
    det_path=None,
    allow_skip=True
):
    """
    Main function to generate pseudo labels using AG Dataset
    """

    # Load config
    cfg_from_file(config_path)

    # Override det_path if specified
    if det_path is not None:
        load_det_path = det_path
    else:
        load_det_path = conf.det_path

    # Initialize AG dataset
    print('Initializing AG Dataset...')
    AG_dataset = AG(
        mode="train",
        datasize=conf.datasize,
        data_path=conf.data_path,
        ws_object_bbox_path=conf.ws_object_bbox_path,
        remove_one_frame_video=False,
        filter_nonperson_box_frame=True,
        filter_small_box=True
    )
    object_classes = AG_dataset.object_classes
    attention_relationships = AG_dataset.attention_relationships
    spatial_relationships = AG_dataset.spatial_relationships
    contacting_relationships = AG_dataset.contacting_relationships
    print(f'Total videos in dataset: {len(AG_dataset)}')

    with open(conf.ws_object_bbox_path, 'rb') as f:
        weak_anno = pickle.load(f)
    vid_dict = generate_video_list(list(weak_anno.keys()))

    # DataLoader (batch_size=1 for video-level processing)
    dataloader = DataLoader(
        AG_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=cuda_collate_fn
    )

    # Initialize detector
    gpu_device = torch.device(f'cuda:{gpu_id}')
    object_detector = detector(
        train=True,
        object_classes=object_classes,
        use_SUPPLY=True,
        conf=conf
    ).to(device=gpu_device)
    object_detector.eval()

    # FasterRCNN for feature extraction
    fasterrcnn_model, transforms = prepare_func()

    # Output annotation
    output_anno = {}

    # Statistics
    stats = {
        'total_videos': 0,
        'total_frames': 0,
        'frames_with_objects': 0,
        'center_with_objects': 0,
        'propagation_success': 0,
        'propagation_skip': 0,
        'propagation_early_stop': 0,
    }

    print('Starting pseudo label generation...')

    for batch in tqdm(dataloader, desc='Processing videos'):

        if conf.is_wks:
            im_data = None
            im_info = None
            gt_boxes = None
            num_boxes = None
        else:
            im_data = batch[0]
            im_info = batch[1]
            gt_boxes = batch[2]
            num_boxes = batch[3]

        gt_annotation = AG_dataset.gt_annotations[batch[4]]
        frame_names = AG_dataset.video_list[batch[4]]
        # vid_dict의 키와 일치하도록 첫 5글자 사용 (예: '001YG')
        vid = frame_names[0][:5]
        frame_list = vid_dict.get(vid, [])

        # frame_list가 비어있으면 스킵
        if len(frame_list) == 0:
            continue

        frame_num = len(frame_list)
        stats['total_videos'] += 1
        stats['total_frames'] += frame_num

        # Compute entry through object_detector
        with torch.no_grad():
            try:
                entry = object_detector(
                    im_data, im_info, gt_boxes, num_boxes,
                    gt_annotation, frame_names,
                    fasterrcnn_model, transforms
                )
            except Exception as e:
                print(f'Error processing video: {e}')
                continue

        if entry is None:
            continue

        # frame_names (weak anno 프레임)을 frame_list (전체 프레임) 내 위치로 매핑
        # entry의 frame_idx는 frame_names 기준이므로, 실제 frame_list 위치로 변환 필요
        entry_idx_to_actual_idx = {}
        for entry_frame_idx, weak_frame_name in enumerate(frame_names):
            # frame_list에서 해당 프레임 찾기
            try:
                actual_idx = frame_list.index(weak_frame_name)
                entry_idx_to_actual_idx[entry_frame_idx] = actual_idx
            except ValueError:
                # 프레임을 찾지 못한 경우 (발생하면 안됨)
                print(f'Warning: {weak_frame_name} not found in frame_list')
                continue

        if len(entry_idx_to_actual_idx) == 0:
            continue

        # 1. det_path에서 모든 프레임의 detection 로드 (bbox, class만 - relation 없음)
        all_frame_detections = load_detections_from_path(frame_list, load_det_path, object_classes)

        # 2. Entry에서 center frame의 objects 추출 (relation 정보 포함)
        center_frame_objects = entry_to_frame_detections(entry, frame_num, object_classes, entry_idx_to_actual_idx)

        # 3. Center frame 인덱스 (weak annotation이 있는 프레임의 실제 위치)
        center_idx = list(entry_idx_to_actual_idx.values())[0]

        # Center frame에 object가 있는지 확인
        if len(center_frame_objects[center_idx]) > 0:
            stats['center_with_objects'] += 1

        # 4. Propagation (center의 relation을 다른 프레임으로 전파)
        # all_frame_detections: 모든 프레임의 detection (IOU 비교용)
        # center_frame_objects[center_idx]: center의 objects with relations (시작점)
        propagated, prop_stats = propagate_relations(
            all_frame_detections, center_frame_objects[center_idx], center_idx, frame_num, iou_threshold, allow_skip
        )
        stats['propagation_success'] += prop_stats['success']
        stats['propagation_skip'] += prop_stats['skip']
        stats['propagation_early_stop'] += prop_stats['early_stop']

        # 결과에서 object가 있는 프레임 수 계산
        for frame_idx in range(frame_num):
            if len(propagated[frame_idx]) > 0:
                stats['frames_with_objects'] += 1

        # PKL 형식으로 변환 및 저장 (frame_list 사용, weak_anno에서 person_info 가져옴)
        video_result = convert_to_pkl_format(
            propagated, frame_list, weak_anno, object_classes,
            attention_relationships, spatial_relationships, contacting_relationships
        )
        output_anno.update(video_result)

    # Print statistics
    print('\n' + '=' * 60)
    print('Generation Complete!')
    print('=' * 60)
    print(f"Total videos: {stats['total_videos']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Frames with objects: {stats['frames_with_objects']} ({100 * stats['frames_with_objects'] / max(1, stats['total_frames']):.1f}%)")
    print(f"Center frames with objects: {stats['center_with_objects']}")
    print(f"Propagation success: {stats['propagation_success']}")
    print(f"Propagation skipped: {stats['propagation_skip']}")
    print(f"Propagation early stopped: {stats['propagation_early_stop']}")
    print(f"Allow skip mode: {allow_skip}")
    print('=' * 60)

    # Save output
    print(f'\nSaving to {output_path}...')
    with open(output_path, 'wb') as f:
        pickle.dump(output_anno, f)
    print('Done!')

    return output_anno, stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/pla_stage_1/glip_ours.yml')
    parser.add_argument('--output', type=str,
                        default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/annotations/weak/pla_stage2_glip.pkl')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--label_num', type=int, default=1, help='Number of GT frames per video')
    parser.add_argument('--det_path', type=str,
                        default='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/PLA_det_ag_class',
                        help='Path to detection results (use full detection for all frames)')
    parser.add_argument('--no_skip', action='store_true',
                        help='Stop propagation when matching fails (default: False, i.e. continue propagation)')

    args = parser.parse_args()

    generate_pseudo_labels(
        config_path=args.config,
        output_path=args.output,
        iou_threshold=args.iou,
        gpu_id=args.gpu,
        label_num=args.label_num,
        det_path=args.det_path,
        allow_skip=not args.no_skip
    )
