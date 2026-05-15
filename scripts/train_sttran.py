import os
import argparse
import json
from lib.config import conf, cfg_from_file

import pdb
"""------------------------------------some settings----------------------------------------"""
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/demo.yml', type=str)
args = parser.parse_args()
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

# Auto-generate timestamped save directory
import time as time_module
timestamp_dir = time_module.strftime("%m%d_%H%M")
# Create descriptive directory name
run_name = f"{timestamp_dir}_{conf.transformer_mode}_{conf.match}"
if getattr(conf, 'unmatched_sampling', False):
    run_name += "_unmatched"

# Update save_path with timestamped subdirectory
base_save_path = conf.save_path
conf.save_path = os.path.join(base_save_path, run_name)
os.makedirs(conf.save_path, exist_ok=True)

print('The CKPT saved here:', conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
print('-------------student model setting------------------')
print(args.cfg_file)
print(conf)
with open(os.path.join(conf.save_path, "configs.json"), 'w') as f:
    json.dump(conf, f)
"""-----------------------------------------------------------------------------------------"""

os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.gpu_id)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log_softmax, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
np.set_printoptions(precision=3)
import time
import pandas as pd
import copy
from tqdm import tqdm
torch.set_num_threads(4)
from tensorboardX import SummaryWriter
import wandb

from lib.loss import (PairAffinityMemoryBank, pair_mask_adaptive_margin_loss)

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from lib.sttran import STTran
from lib.assign_pseudo_label import prepare_func
from lib.transition_module import transition_module
from lib.ults.track import track, track_diff, track_iou, track_diff_iou
from lib.ults.init_teacher_model import init_teacher_model


def to_scalar(v):
    if hasattr(v, 'item'):
        return v.item()
    return float(v)

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, ws_object_bbox_path=conf.ws_object_bbox_path, remove_one_frame_video=conf.remove_one_frame_video,
                      filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=conf.num_workers,batch_size = 1,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, ws_object_bbox_path=None, remove_one_frame_video=True,
                     filter_nonperson_box_frame=True, filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=conf.num_workers, batch_size = 1,
                                              collate_fn=cuda_collate_fn, pin_memory=False)
gpu_device = torch.device("cuda")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, conf=conf).to(device=gpu_device)
object_detector.eval()

if conf.union_box_feature:
    faset_rcnn_model, transforms = prepare_func()
else:
    faset_rcnn_model = None
    transforms = None

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               transformer_mode=conf.transformer_mode,
               is_wks=conf.is_wks,
               feat_dim=conf.feat_dim,
               obj_dim=conf.obj_dim,
               conf=conf).to(device=gpu_device)
print("create student model successfully")
print('*'*50)

# Initialize Memory Bank for negative embeddings
neg_memory_bank = PairAffinityMemoryBank(max_size=1024, emb_dim=256)
print("Memory Bank initialized (size=1024)")

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
use_uncertainty = False
if conf.teacher_mode_cfg is None:
    print("Do not need to create teacher model")
    print('*'*50)
else:
    t_model = init_teacher_model(conf.teacher_mode_cfg, AG_dataset_train, gpu_device)

    print("create teacher model successfully")
    print('*'*50)

    # switch dropout on in inference stage
    t_model.eval()  # TODO

    if use_uncertainty:
        t_model.apply(apply_dropout)  # TODO

if conf.ckpt is None:
    print('Do not need to load CKPT')
    start_epoch = 0
else:
    ckpt = torch.load(conf.ckpt, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    print('CKPT {} is loaded'.format(conf.ckpt))
    left_pos = conf.ckpt.rfind('_')
    right_pos = conf.ckpt.rfind('.')
    start_epoch = int(conf.ckpt[left_pos+1:right_pos]) + 1

evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='with',
    conf=conf)

# evaluator2 = BasicSceneGraphEvaluator(
#     mode=conf.mode,
#     AG_object_classes=AG_dataset_train.object_classes,
#     AG_all_predicates=AG_dataset_train.relationship_classes,
#     AG_attention_predicates=AG_dataset_train.attention_relationships,
#     AG_spatial_predicates=AG_dataset_train.spatial_relationships,
#     AG_contacting_predicates=AG_dataset_train.contacting_relationships,
#     iou_threshold=0.5,
#     constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset_train.object_classes,
    AG_all_predicates=AG_dataset_train.relationship_classes,
    AG_attention_predicates=AG_dataset_train.attention_relationships,
    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
    iou_threshold=0.5,
    constraint='no',
    conf=conf)

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    ce_loss_none = nn.CrossEntropyLoss(reduction="none")
    # bce_loss = nn.BCEWithLogitsLoss()
    bce_loss = nn.BCELoss()
    bce_loss_none = nn.BCELoss(reduction="none")
    # if conf.loss == 'KL':
        # kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss = nn.KLDivLoss(reduction="sum")
    kl_loss_none = nn.KLDivLoss(reduction="none")
    # elif conf.loss == 'L1':
    L1_loss = nn.L1Loss()
    # elif conf.loss == 'L2':
    L2_loss = nn.MSELoss()
    # Pair Affinity losses (weak supervision)
    margin_ranking_loss = nn.MarginRankingLoss(margin=1.0)  # Logit space: larger margin
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()
    # Pair Affinity losses (weak supervision)
    margin_ranking_loss = nn.MarginRankingLoss(margin=1.0)  # Logit space: larger margin


softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()

if conf.transition_module:
    trans_module = transition_module().to(device=gpu_device)
else:
    trans_module = None

# optimizer
if conf.optimizer == 'adamw':
    # optimizer = AdamW(model.parameters(), lr=conf.lr)
    if trans_module is None:
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif trans_module is not None:
        optimizer = AdamW([{'params': model.parameters()}, {'params': trans_module.parameters(), 'lr': conf.t_lr}], lr=conf.lr)
        # optimizer = AdamW([{'params': model.parameters()}, {'params': trans_module.parameters(), 'lr': conf.lr}], lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

#scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
scheduler = CosineAnnealingLR(optimizer, T_max=conf.nepoch, eta_min=1e-7)

# Mixed Precision Training (FP16) - saves ~40-50% GPU memory
# NOTE: Disabled due to dtype compatibility issues with current model architecture
# The model outputs and intermediate tensors need consistent dtype handling for AMP
scaler = torch.cuda.amp.GradScaler()
use_amp = False  # Set to True to enable AMP (requires fixing dtype issues in transformer layers)
print(f"Mixed Precision Training (AMP): {'Enabled' if use_amp else 'Disabled'}")

writer = SummaryWriter(conf.tensorboard_name)

# Initialize wandb
timestamp = time.strftime("%m%d_%H%M")  
wandb.init(
    project="WS-DSGG",  
    name=f"{timestamp}",  # Run 이름
    config=conf,  # Configuration 저장
    tags=['sttran', 'stage2' if conf.teacher_mode_cfg else 'stage1', conf.transformer_mode, conf.mode, conf.match],  # Tags for filtering
)

# some parameters
tr = []

test_res = {}

save_loss = {}

save_epoch = 1000

# Track best model based on with_avg + no_avg
best_score = -1.0
best_epoch = -1

# gt_uncertainty_stats = [[] for i in range(11)]
# gt_nums_stats = [0 for i in range(11)]
# other_uncertainty_stats = [[] for i in range(11)]
# other_nums_stats = [0 for i in range(11)]

actual_end = conf.stop_epoch if conf.stop_epoch is not None else conf.nepoch
for epoch in range(start_epoch, actual_end):
    model.train() # TODO
    object_detector.is_train = True
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)

    # Object distribution statistics
    object_count_stats = {}

    alpha1_cnt = 0
    alpha2_cnt = 0
    loss_cnt_tr = []

    pseudo_obj_list = []
    found_gt__list = []
    unfound_gt__list = []
    with tqdm(total=len(dataloader_train)) as t:
        for b in range(len(dataloader_train)):
            data = next(train_iter)

            if conf.is_wks:
                im_data = None
                im_info = None
                gt_boxes = None
                num_boxes = None
            else:
                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_train.gt_annotations[data[4]]
            real_gt_annotation = AG_dataset_train.real_gt_annotations[data[4]]
            frame_names = AG_dataset_train.video_list[data[4]]

            # prevent gradients to FasterRCNN
            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, frame_names, faset_rcnn_model, transforms)
                t_entry = copy.deepcopy(entry)

            # Track object distribution statistics
            if entry is None:
                num_objects = 0
            else:
                num_objects = len(entry['pair_idx'])  # Number of person-object pairs
            object_count_stats[num_objects] = object_count_stats.get(num_objects, 0) + 1

            if entry != None:
                if conf.teacher_mode_cfg is not None:
                    with torch.no_grad():
                        if use_uncertainty:
                            t_pred, attention_std, spatial_std, contact_std = uncertainty(t_model, t_entry, 10)
                        else:
                            t_pred = t_model(t_entry) # TODO

                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(entry) # TODO

                attention_distribution = pred["attention_distribution"]
                spatial_distribution = pred["spatial_distribution"]
                contact_distribution = pred["contacting_distribution"]

                # Attention predictions (softmax for single-label)
                attention_probs = torch.softmax(attention_distribution, dim=1)
                attention_preds = torch.argmax(attention_probs, dim=1)

                # Spatial predictions (sigmoid for multi-label)
                spatial_probs = torch.sigmoid(spatial_distribution)

                # Contact predictions (sigmoid for multi-label)
                contact_probs = torch.sigmoid(contact_distribution)

                object_label = pred['labels']
                attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device)
                # 对于[[1]]这种形状，只squeeze一维
                if attention_label.shape[0] > 1:
                    attention_label.squeeze_()
                else:
                    attention_label.squeeze_(1)
                # attention_label: 一维tensor，每个值对应每对people-object的关系类别，整体like tensor([2, 1, 1, 2, 0, 0], device='cuda:0')
                if not conf.bce_loss:
                    # multi-label margin loss or adaptive loss
                    # spatial_label/contact_label: 二维tensor，其中每个一维tensor对应每对people-object的关系类别
                    # 每个一维tensor like tensor([ 2,  4, -1, -1, -1, -1], device='cuda:0')，-1之前的表示gt的关系
                    spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
                    contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"])):
                        spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                        contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

                else:
                    # bce loss
                    # spatial_label/contact_label: 二维tensor，其中每个一维tensor对应每对people-object的关系类别
                    # 每个一维tensor like tensor([ 0,  0,  1,  0,  1,  0], device='cuda:0')，1表示gt的关系，其他表示没有gt关系
                    spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
                    contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"])):
                        spatial_label[i, pred["spatial_gt"][i]] = 1
                        contact_label[i, pred["contacting_gt"][i]] = 1

                # 确定soft target和hard target的比例
                # 只需要soft target或者只需要hard target时，设定alpha=0或1即可
                if conf.temperature is None or conf.temperature == 1:
                    temperature = 1
                else:
                    temperature = conf.temperature

                losses = {}
                if conf.alpha is None or conf.alpha == 0:
                    alpha = 0
                else:
                    alpha = conf.alpha
                
                if 'rel_gt' not in pred:
                    gt_rel = torch.ones(len(pred['spatial_gt']), dtype=torch.bool, device=spatial_label.device)
                else:
                    gt_rel = pred['rel_gt']

                # Extract negative_mask (True = exclude from loss, False = include in loss)
                if 'negative_mask' not in pred:
                    negative_mask = torch.zeros(len(pred['spatial_gt']), dtype=torch.bool, device=spatial_label.device)
                else:
                    negative_mask = pred['negative_mask']

                # Create mask for loss calculation (True = use for loss)
                use_for_loss = ~negative_mask

                if alpha != 0:
                    # 需要soft label计算蒸馏损失的情况，否则只需要hard label
                    if conf.teacher_mode_cfg is not None:                            
                        
                        student_object_distribution = pred['distribution']
                        student_attention_distribution = pred["attention_distribution"]
                        student_spatial_distribution = pred["spatial_distribution"]
                        student_contact_distribution = pred["contacting_distribution"]

                        teacher_object_distribution = t_pred['distribution']
                        teacher_attention_distribution = t_pred["attention_distribution"]
                        teacher_spatial_distribution = t_pred["spatial_distribution"]
                        teacher_contact_distribution = t_pred["contacting_distribution"]

                        if conf.label_fusion_strategy == 0:
                            fusion_spatial_distribution = teacher_spatial_distribution * alpha + spatial_label * (1-alpha)
                            fusion_contact_distribution = teacher_contact_distribution * alpha + contact_label * (1-alpha)
                        elif conf.label_fusion_strategy == 1:
                            pred_spatial_label, pred_contact_label = trans_module(teacher_spatial_distribution, teacher_contact_distribution, pred['obj_labels'])

                            # Negative sampling 시 positive object만 track_iou에 사용
                            # negative_mask는 pair 기준 [num_pairs], labels/boxes는 human 포함 [num_objects]
                            t_neg_mask = t_pred.get('negative_mask', None)
                            if t_neg_mask is not None and t_neg_mask.any():
                                pos_mask = ~t_neg_mask  # [num_pairs] = [num_non_human_objects]
                                pos_indices = torch.where(pos_mask)[0]

                                # human 제외한 labels, boxes 추출
                                non_human_mask = t_pred['labels'] != 1
                                non_human_labels = t_pred['labels'][non_human_mask]
                                non_human_boxes = t_pred['boxes'][non_human_mask, 1:5]

                                # positive만 필터링
                                pos_labels = non_human_labels[pos_mask]
                                pos_im_idx = t_pred['im_idx'][pos_mask]
                                pos_boxes = non_human_boxes[pos_mask]

                                filtered_pseudo_id, filtered_transition_id = track_iou(pos_labels, pos_im_idx, pos_boxes, 0.5)

                                # 필터링된 인덱스를 원본 인덱스로 매핑
                                pseudo_id = [pos_indices[i].item() for i in filtered_pseudo_id]
                                transition_id = [pos_indices[i].item() for i in filtered_transition_id]
                            else:
                                pseudo_id, transition_id = track_iou(t_pred['labels'], t_pred['im_idx'], t_pred['boxes'][:, 1:5], 0.5)

                            spatial_relation_loss_pred = kl_loss(F.log_softmax(teacher_spatial_distribution[pseudo_id], dim=1), F.softmax(pred_spatial_label[transition_id], dim=1))
                            contact_relation_loss_pred = kl_loss(F.log_softmax(teacher_contact_distribution[pseudo_id], dim=1), F.softmax(pred_contact_label[transition_id], dim=1))

                            losses['spatial_relation_loss_pred'] = spatial_relation_loss_pred
                            losses['contact_relation_loss_pred'] = contact_relation_loss_pred
                            alpha1 = 2 - 2 * F.sigmoid(spatial_relation_loss_pred).detach()
                            alpha2 = 2 - 2 * F.sigmoid(contact_relation_loss_pred).detach()
                            fusion_spatial_distribution = teacher_spatial_distribution * alpha1 + spatial_label * (1-alpha1)
                            fusion_contact_distribution = teacher_contact_distribution * alpha2 + contact_label * (1-alpha2)
                            alpha1_cnt += alpha1.item()
                            alpha2_cnt += alpha2.item()

                        fusion_spatial_distribution[gt_rel] = spatial_label[gt_rel]
                        fusion_contact_distribution[gt_rel] = contact_label[gt_rel]
                        spatial_label = fusion_spatial_distribution
                        contact_label = fusion_contact_distribution
                
                # Attention loss (using mask)
                if conf.loss == 'KL' and alpha != 0 and conf.teacher_mode_cfg is not None:
                    # KL loss with teacher attention distribution
                    # Note: teacher_attention_distribution is raw logits, need softmax to convert to probabilities
                    fusion_attention_distribution = F.softmax(teacher_attention_distribution, dim=1)
                    # GT frames: replace with one-hot hard label
                    gt_attention_onehot = F.one_hot(attention_label[gt_rel], num_classes=3).float()
                    fusion_attention_distribution[gt_rel] = gt_attention_onehot

                    # KL loss: pred (log_softmax) vs target (softmax from teacher or one-hot)
                    attention_loss_all = kl_loss_none(F.log_softmax(attention_distribution, dim=1),
                                                      fusion_attention_distribution).sum(dim=1)
                    if use_for_loss.sum() > 0:
                        losses["attention_relation_loss"] = (attention_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                    else:
                        losses["attention_relation_loss"] = attention_loss_all.sum() * 0.0
                else:
                    # Original CE loss (when not using KL or no teacher)
                    attention_loss_all = ce_loss_none(attention_distribution, attention_label)
                    if use_for_loss.sum() > 0:
                        losses["attention_relation_loss"] = (attention_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                    else:
                        losses["attention_relation_loss"] = attention_loss_all.sum() * 0.0  # Zero loss if no valid samples

                if not conf.bce_loss:
                    # MLM loss doesn't support reduction='none', so we use masking differently
                    # Filter out negative samples before loss calculation
                    if use_for_loss.sum() > 0:
                        losses["spatial_relation_loss"] = mlm_loss(spatial_distribution[use_for_loss], spatial_label[use_for_loss])
                        losses["contact_relation_loss"] = mlm_loss(contact_distribution[use_for_loss], contact_label[use_for_loss])
                    else:
                        losses["spatial_relation_loss"] = torch.tensor(0.0, device=spatial_distribution.device)
                        losses["contact_relation_loss"] = torch.tensor(0.0, device=spatial_distribution.device)

                else:
                    if conf.loss == 'KL':
                        # # [Original] Softmax KL loss with masking (treats as single-label)
                        # spatial_loss_all = kl_loss_none(F.log_softmax(spatial_distribution, dim=1),
                        #                                 F.softmax(spatial_label, dim=1)).sum(dim=1)
                        # contact_loss_all = kl_loss_none(F.log_softmax(contact_distribution, dim=1),
                        #                                 F.softmax(contact_label, dim=1)).sum(dim=1)
                        # if use_for_loss.sum() > 0:
                        #     losses["spatial_relation_loss"] = (spatial_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                        #     losses["contact_relation_loss"] = (contact_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                        # else:
                        #     losses["spatial_relation_loss"] = spatial_loss_all.sum() * 0.0
                        #     losses["contact_relation_loss"] = contact_loss_all.sum() * 0.0

                        # [New] Binary KL loss (per-class independent, proper multi-label)
                        # Note: spatial_distribution and contact_distribution are already sigmoid outputs from model
                        eps = 1e-7

                        # Spatial: binary KL (model output is already sigmoid)
                        pred_spatial = torch.clamp(spatial_distribution, eps, 1-eps)
                        target_spatial = torch.clamp(spatial_label, eps, 1-eps)
                        spatial_loss_all = (target_spatial * torch.log(target_spatial / pred_spatial) +
                                            (1 - target_spatial) * torch.log((1 - target_spatial) / (1 - pred_spatial))).sum(dim=1)

                        # Contact: binary KL (model output is already sigmoid)
                        pred_contact = torch.clamp(contact_distribution, eps, 1-eps)
                        target_contact = torch.clamp(contact_label, eps, 1-eps)
                        contact_loss_all = (target_contact * torch.log(target_contact / pred_contact) +
                                            (1 - target_contact) * torch.log((1 - target_contact) / (1 - pred_contact))).sum(dim=1)

                        if use_for_loss.sum() > 0:
                            losses["spatial_relation_loss"] = (spatial_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                            losses["contact_relation_loss"] = (contact_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                        else:
                            losses["spatial_relation_loss"] = spatial_loss_all.sum() * 0.0
                            losses["contact_relation_loss"] = contact_loss_all.sum() * 0.0

                    elif conf.loss == 'L1':
                        # L1 loss - filter before calculation
                        if use_for_loss.sum() > 0:
                            losses["spatial_relation_loss"] = L1_loss(spatial_distribution[use_for_loss], spatial_label[use_for_loss])
                            losses["contact_relation_loss"] = L1_loss(contact_distribution[use_for_loss], contact_label[use_for_loss])
                        else:
                            losses["spatial_relation_loss"] = torch.tensor(0.0, device=spatial_distribution.device)
                            losses["contact_relation_loss"] = torch.tensor(0.0, device=spatial_distribution.device)

                    elif conf.loss == 'L2':
                        # L2 loss - filter before calculation
                        if use_for_loss.sum() > 0:
                            losses["spatial_relation_loss"] = L2_loss(spatial_distribution[use_for_loss], spatial_label[use_for_loss])
                            losses["contact_relation_loss"] = L2_loss(contact_distribution[use_for_loss], contact_label[use_for_loss])
                        else:
                            losses["spatial_relation_loss"] = torch.tensor(0.0, device=spatial_distribution.device)
                            losses["contact_relation_loss"] = torch.tensor(0.0, device=spatial_distribution.device)

                    elif conf.loss == 'BCE':
                        # BCE loss with masking
                        spatial_loss_all = bce_loss_none(spatial_distribution, spatial_label).mean(dim=1)
                        contact_loss_all = bce_loss_none(contact_distribution, contact_label).mean(dim=1)
                        if use_for_loss.sum() > 0:
                            losses["spatial_relation_loss_BCE"] = (spatial_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                            losses["contact_relation_loss_BCE"] = (contact_loss_all * use_for_loss.float()).sum() / use_for_loss.sum()
                        else:
                            losses["spatial_relation_loss_BCE"] = spatial_loss_all.sum() * 0.0
                            losses["contact_relation_loss_BCE"] = contact_loss_all.sum() * 0.0

                # =============================================
                # PA Loss: balanced distance-weighted BCE (single fixed loss)
                # Soft target = α · pseudo_gt + (1 − α) · sigmoid(teacher_logit),
                # where α decays with temporal distance from the annotated frame.
                # Positive (target ≥ 0.5) and negative (target < 0.5) groups are
                # balanced 1:1 in the final BCE.
                # =============================================
                if (negative_mask.sum() > 0 and use_for_loss.sum() > 0
                        and "pair_affinity" in pred
                        and conf.teacher_mode_cfg is not None
                        and "pair_affinity" in t_pred):
                    logits = pred["pair_affinity"].squeeze()
                    bce_weight  = getattr(conf, 'pa_weight', 1.0)
                    alpha_power = getattr(conf, 'pa_alpha_power', 3.0)

                    num_frames_pa   = int(pred['im_idx'].max().item()) + 1
                    center_frame_pa = num_frames_pa // 2
                    max_distance_pa = num_frames_pa / 2.0
                    distances_pa    = torch.abs(pred['im_idx'].float() - center_frame_pa)
                    alpha_pa        = (1.0 - distances_pa / max_distance_pa).clamp(min=0.0) ** alpha_power
                    t_sigmoid_pa    = torch.sigmoid(t_pred["pair_affinity"].squeeze())
                    pseudo_gt       = use_for_loss.float()
                    target_pa       = alpha_pa * pseudo_gt + (1.0 - alpha_pa) * t_sigmoid_pa

                    pos_target_mask = target_pa >= 0.5
                    neg_target_mask = target_pa < 0.5
                    if pos_target_mask.sum() > 0 and neg_target_mask.sum() > 0:
                        pos_bce = F.binary_cross_entropy_with_logits(logits[pos_target_mask], target_pa[pos_target_mask])
                        neg_bce = F.binary_cross_entropy_with_logits(logits[neg_target_mask], target_pa[neg_target_mask])
                        losses["pa_loss"] = bce_weight * (pos_bce + neg_bce) / 2

                # =============================================
                # PAM Loss: adaptive-margin triplet over pair_emb @ pair_emb^T
                # (single fixed loss)
                # =============================================
                if negative_mask.sum() > 0 and use_for_loss.sum() > 0 and "all_pair_mask_logits" in pred:
                    pam_weight = getattr(conf, 'pam_weight', 0.3)
                    pam_margin = getattr(conf, 'pam_margin', 1.0)
                    propagate_margin = getattr(conf, 'propagate_margin', 0.3)
                    mixed_margin = (pam_margin + propagate_margin) / 2

                    all_pair_mask_logits = pred["all_pair_mask_logits"]
                    if len(all_pair_mask_logits) > 0:
                        pair_mask_logit = all_pair_mask_logits[-1]
                        num_windows, L_dim, _ = pair_mask_logit.shape

                        im_idx = pred['im_idx']

                        # Build target_logit for pos/neg determination
                        if conf.teacher_mode_cfg is not None and "pair_affinity" in t_pred:
                            alpha_power = getattr(conf, 'pa_alpha_power', 3.0)
                            num_frames = int(im_idx.max().item()) + 1
                            center_frame = num_frames // 2
                            max_distance = num_frames / 2.0

                            distances = torch.abs(im_idx.float() - center_frame)
                            alpha = (1.0 - distances / max_distance).clamp(min=0.0) ** alpha_power

                            t_sigmoid = torch.sigmoid(t_pred["pair_affinity"].squeeze())
                            pseudo_gt = use_for_loss.float()

                            target_logit = alpha * pseudo_gt + (1.0 - alpha) * t_sigmoid
                            positive_mask_flat = (target_logit >= 0.5).float()
                        else:
                            positive_mask_flat = (~negative_mask).float()

                        gt_mask_flat = gt_rel.float()

                        # Build sliding window masks
                        if conf.transformer_mode == 'wk':
                            num_frames = int(im_idx.max().item()) + 1

                            positive_mask = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device)
                            gt_mask = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device, dtype=torch.bool)
                            valid_mask = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device, dtype=torch.bool)
                            target_logit_padded = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device)

                            for j in range(num_windows):
                                frame_t_pairs = (im_idx == j)
                                frame_t1_pairs = (im_idx == j + 1)
                                num_t = frame_t_pairs.sum().item()
                                num_t1 = frame_t1_pairs.sum().item()

                                if num_t > 0:
                                    positive_mask[j, :num_t] = positive_mask_flat[frame_t_pairs]
                                    gt_mask[j, :num_t] = gt_rel[frame_t_pairs]
                                    valid_mask[j, :num_t] = True
                                    if conf.teacher_mode_cfg is not None:
                                        target_logit_padded[j, :num_t] = target_logit[frame_t_pairs]
                                if num_t1 > 0:
                                    positive_mask[j, num_t:num_t+num_t1] = positive_mask_flat[frame_t1_pairs]
                                    gt_mask[j, num_t:num_t+num_t1] = gt_rel[frame_t1_pairs]
                                    valid_mask[j, num_t:num_t+num_t1] = True
                                    if conf.teacher_mode_cfg is not None:
                                        target_logit_padded[j, num_t:num_t+num_t1] = target_logit[frame_t1_pairs]

                        else:
                            positive_mask = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device)
                            gt_mask = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device, dtype=torch.bool)
                            valid_mask = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device, dtype=torch.bool)
                            target_logit_padded = torch.zeros(num_windows, L_dim, device=pair_mask_logit.device)

                            for frame_idx in range(num_windows):
                                frame_pairs = (im_idx == frame_idx)
                                num_pairs = frame_pairs.sum().item()
                                if num_pairs > 0:
                                    positive_mask[frame_idx, :num_pairs] = positive_mask_flat[frame_pairs]
                                    gt_mask[frame_idx, :num_pairs] = gt_rel[frame_pairs]
                                    valid_mask[frame_idx, :num_pairs] = True
                                    if conf.teacher_mode_cfg is not None:
                                        target_logit_padded[frame_idx, :num_pairs] = target_logit[frame_pairs]

                        valid_mask_2d = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
                        gt_outer = gt_mask.unsqueeze(2) & gt_mask.unsqueeze(1) & valid_mask_2d
                        prop_outer = (~gt_mask).unsqueeze(2) & (~gt_mask).unsqueeze(1) & valid_mask_2d
                        mixed_outer = ~gt_outer & ~prop_outer & valid_mask_2d
                        pos_mask_2d = (positive_mask.unsqueeze(2) > 0) & (positive_mask.unsqueeze(1) > 0)

                        losses["pam_loss"] = pam_weight * pair_mask_adaptive_margin_loss(
                            pair_mask_logit, pos_mask_2d & valid_mask_2d,
                            confidence=target_logit_padded,
                            base_margin=pam_margin, valid_mask=valid_mask_2d,
                        )

                optimizer.zero_grad()
                loss = sum(losses.values())

                # Ensure loss is float32 for stable backward pass (required for AMP)
                if loss.dtype != torch.float32:
                    loss = loss.float()

                # Mixed precision backward pass
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
                    optimizer.step()

                for k in losses.keys():
                    writer.add_scalar(k, losses[k], epoch * len(dataloader_train) + b)

                # Log to wandb
                wandb_log = {f"train/{k}": v.item() for k, v in losses.items()}
                wandb_log["train/total_loss"] = loss.item()
                wandb_log["train/epoch"] = epoch
                wandb_log["train/batch"] = b
                wandb.log(wandb_log, step=epoch * len(dataloader_train) + b)

                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
                loss_cnt_tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

            if b % save_epoch == 0 and b >= save_epoch:
                time_per_batch = (time.time() - start) / save_epoch
                t.write("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                    time_per_batch, len(dataloader_train) * time_per_batch / 60))

                mn = pd.concat(tr[-save_epoch:], axis=1).mean(1) # TODO
                t.write(str(mn))
                start = time.time()

            t.set_description(desc="Epoch {} ".format(epoch))
            # t.set_postfix(steps=step, loss=loss.data.item())
            t.update(1)


    save_loss[str(epoch)] = {}
    loss_all = pd.concat(loss_cnt_tr, axis=1).mean(1)
    for loss_name in loss_all.keys():
        save_loss[str(epoch)][loss_name] = loss_all[loss_name]
    save_loss[str(epoch)]['alpha1'] = alpha1_cnt / b
    save_loss[str(epoch)]['alpha2'] = alpha2_cnt / b
    with open(os.path.join(conf.save_path, "save_loss_{}.json".format(epoch)), 'w') as f:
        json.dump(save_loss, f)

    # Log epoch-level average losses to wandb
    # Use step that's after training steps to avoid "step going backwards" error
    epoch_avg_step = (epoch + 1) * len(dataloader_train) - 1
    wandb_epoch_loss = {f"epoch_avg/{k}": v for k, v in loss_all.items()}
    wandb_epoch_loss["epoch_avg/alpha1"] = alpha1_cnt / b
    wandb_epoch_loss["epoch_avg/alpha2"] = alpha2_cnt / b
    wandb_epoch_loss["epoch_avg/epoch"] = epoch
    wandb.log(wandb_epoch_loss, step=epoch_avg_step)

    # Object distribution statistics
    print("\n" + "="*60)
    print(f"EPOCH {epoch} - Object Distribution Statistics")
    print("="*60)
    total_samples = sum(object_count_stats.values())
    samples_0 = object_count_stats.get(0, 0)
    samples_1 = object_count_stats.get(1, 0)
    samples_2plus = sum(object_count_stats.get(i, 0) for i in range(2, max(object_count_stats.keys())+1)) if len(object_count_stats) > 0 else 0

    print(f"Total samples:              {total_samples}")
    print(f"  0 objects (NOT trained):  {samples_0:>6} ({samples_0/total_samples*100:5.1f}%)")
    print(f"  1 object (BROKEN attn):   {samples_1:>6} ({samples_1/total_samples*100:5.1f}%)")
    print(f"  2+ objects (NORMAL attn): {samples_2plus:>6} ({samples_2plus/total_samples*100:5.1f}%)")
    print(f"Wasted (0-1 objects):       {samples_0+samples_1:>6} ({(samples_0+samples_1)/total_samples*100:5.1f}%)")
    print("="*60 + "\n")

    # Only run evaluation if epoch >= start_eval_epoch
    if epoch >= conf.start_eval_epoch:
        model.eval()
        object_detector.is_train = False

        # Test set object distribution statistics
        test_object_count_stats = {}  # Per-video (entry) statistics
        test_frame_object_stats = {}  # Per-frame statistics

        with torch.no_grad():
            with tqdm(total=len(dataloader_test)) as t:
                for b in range(len(dataloader_test)):
                    data = next(test_iter)

                    im_data = copy.deepcopy(data[0].cuda(0))
                    im_info = copy.deepcopy(data[1].cuda(0))
                    gt_boxes = copy.deepcopy(data[2].cuda(0))
                    num_boxes = copy.deepcopy(data[3].cuda(0))
                    gt_annotation = AG_dataset_test.gt_annotations[data[4]]
                    frame_names = AG_dataset_test.video_list[data[4]]
                    entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, frame_names, faset_rcnn_model, transforms)

                    # Track test object distribution (per-video)
                    if entry is None:
                        num_objects = 0
                    else:
                        num_objects = len(entry['pair_idx'])
                    test_object_count_stats[num_objects] = test_object_count_stats.get(num_objects, 0) + 1

                    # Track per-frame object distribution
                    if entry is not None and 'im_idx' in entry:
                        # im_idx indicates which frame each pair belongs to
                        im_idx = entry['im_idx'].cpu().numpy()
                        unique_frames = set(im_idx)

                        for frame_idx in unique_frames:
                            # Count objects in this frame
                            num_objects_in_frame = (im_idx == frame_idx).sum()
                            test_frame_object_stats[num_objects_in_frame] = test_frame_object_stats.get(num_objects_in_frame, 0) + 1
                    elif entry is None:
                        # If entry is None, we don't know how many frames, so skip frame-level stats
                        pass

                    if entry != None:
                        pred = model(entry)
                    else:
                        pred = {}
                    # evaluator.evaluate_scene_graph(gt_annotation, pred)
                    evaluator1.evaluate_scene_graph(gt_annotation, dict(pred), frame_names)
                    # evaluator2.evaluate_scene_graph(gt_annotation, dict(pred))
                    evaluator3.evaluate_scene_graph(gt_annotation, dict(pred), frame_names)
                    t.update(1)
                t.write('-----------')
        score = np.mean(evaluator1.result_dict[conf.mode + "_recall"][20])
        evaluator1.print_stats()

        # Test set object distribution statistics
        print("\n" + "="*60)
        print(f"EPOCH {epoch} - TEST Set Object Distribution")
        print("="*60)
        total_test_samples = sum(test_object_count_stats.values())
        test_samples_0 = test_object_count_stats.get(0, 0)
        test_samples_1 = test_object_count_stats.get(1, 0)
        test_samples_2plus = sum(test_object_count_stats.get(i, 0) for i in range(2, max(test_object_count_stats.keys())+1)) if len(test_object_count_stats) > 0 else 0

        print(f"Total test samples:         {total_test_samples}")
        print(f"  0 objects (skipped):      {test_samples_0:>6} ({test_samples_0/total_test_samples*100:5.1f}%)")
        print(f"  1 object (BROKEN attn):   {test_samples_1:>6} ({test_samples_1/total_test_samples*100:5.1f}%)")
        print(f"  2+ objects (NORMAL attn): {test_samples_2plus:>6} ({test_samples_2plus/total_test_samples*100:5.1f}%)")
        print("="*60 + "\n")

        # save res``
        with_res = evaluator1.save_stats()
        # semi_res = evaluator2.save_stats()
        no_res = evaluator3.save_stats()

        # Calculate average of R@10, R@20, R@50
        with_avg = np.mean([with_res['R@10'], with_res['R@20'], with_res['R@50']])
        no_avg = np.mean([no_res['R@10'], no_res['R@20'], no_res['R@50']])
        with_res['R@avg'] = with_avg
        no_res['R@avg'] = no_avg

        # Calculate combined score for best model selection
        current_score = with_avg + no_avg

        # Save best model only
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            if trans_module is not None:
                torch.save({"state_dict": model.state_dict(), "state_dict_3": trans_module.state_dict()},
                          os.path.join(conf.save_path, "best_model.tar"))
            else:
                torch.save({"state_dict": model.state_dict()},
                          os.path.join(conf.save_path, "best_model.tar"))
            # Log best metrics to wandb summary
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_score"] = best_score
            for k, v in with_res.items():
                wandb.run.summary[f"best_with_{k}"] = to_scalar(v)
            for k, v in no_res.items():
                wandb.run.summary[f"best_no_{k}"] = to_scalar(v)
            print("*" * 40)
            print(f"NEW BEST MODEL saved at epoch {epoch}!")
            print(f"Best score: {best_score:.4f} (with_avg: {with_avg:.4f} + no_avg: {no_avg:.4f})")
            print("*" * 40)
        else:
            print(f"Current score: {current_score:.4f} (Best: {best_score:.4f} at epoch {best_epoch})")

        # res = {'with': with_res, 'semi': semi_res, 'no': no_res}
        res = {'with': with_res, 'no': no_res}

        test_res['epoch' + str(epoch)] = res

        for k in with_res.keys():
            writer.add_scalar('with' + k, with_res[k], epoch)
        # for k in semi_res.keys():
        #     writer.add_scalar('semi' + k, semi_res[k], epoch)
        for k in no_res.keys():
            writer.add_scalar('no' + k, no_res[k], epoch)

        # Log evaluation results to wandb
        # Use step that's after training steps to avoid "step going backwards" error
        eval_step = (epoch + 1) * len(dataloader_train)
        wandb_eval_log = {}
        for k, v in with_res.items():
            wandb_eval_log[f"eval/with_{k}"] = to_scalar(v)
        for k, v in no_res.items():
            wandb_eval_log[f"eval/no_{k}"] = to_scalar(v)
        wandb_eval_log["eval/epoch"] = epoch
        wandb_eval_log["eval/score"] = score
        wandb_eval_log["eval/current_score"] = current_score
        wandb_eval_log["eval/best_score"] = best_score
        wandb_eval_log["eval/best_epoch"] = best_epoch
        wandb.log(wandb_eval_log, step=eval_step)

        with open(os.path.join(conf.save_path, "save_res_{}.json".format(epoch)), 'w') as f:
            json.dump(test_res, f)

        evaluator1.reset_result()
        # evaluator2.reset_result()
        evaluator3.reset_result()
        scheduler.step(score)
    else:
        print(f"Skipping evaluation for epoch {epoch} (start_eval_epoch={conf.start_eval_epoch})")

# Training completed
print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
if best_epoch >= 0:
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Model path: {os.path.join(conf.save_path, 'best_model.tar')}")
else:
    print("No model was saved (evaluation was skipped for all epochs)")
print("="*60 + "\n")

# Finish wandb run
wandb.finish()


