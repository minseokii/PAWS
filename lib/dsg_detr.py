"""
DSG-DETR with PAM (Pair Affinity Modulation) integration for TRKT/PLA framework.

Ported from NL-VSGG/lib/dsg_detr.py with the following modifications:
- PAM-aware EncoderLayer (batch_first=True)
- ObjectClassifier adapted from TRKT/PLA sttran.py (obj_dim, is_wks, wv_dir)
- STTran class with PAM integration in spatial (frame) and temporal (object class) grouping
- obj_embed/obj_embed2 use len(obj_classes) (TRKT/PLA convention)
- subj_fc/obj_fc use configurable obj_dim
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from lib.word_vectors import obj_edge_vectors
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from lib.extract_bbox_features import extract_feature_given_bbox_base_feat_torch


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PositionalEncoding(nn.Module):
    """DSG-DETR style sinusoidal positional encoding (batch_first=True: [B, L, D])"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            indices: optional positional indices per batch item
        """
        if indices is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class PAMEncoderLayer(nn.Module):
    """Transformer Encoder Layer with PAM (Pair Affinity Modulation) support.

    batch_first=True convention: inputs/outputs are [B, L, D].
    Manual Q/K/V projection enables pair_emb-based attention modulation.
    """

    def __init__(self, d_model=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, pair_emb_dim=128, pam=True):
        super().__init__()
        self.pam = pam
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # Manual Q/K/V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # MLP for updating pair embedding (residual)
        self.pair_emb_update_mlp = nn.Sequential(
            nn.Linear(d_model, pair_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(pair_emb_dim),
            nn.Linear(pair_emb_dim, pair_emb_dim)
        )

        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, pair_emb=None):
        """
        Args:
            src: [B, L, D]
            src_key_padding_mask: [B, L] (True = padded position)
            pair_emb: [B, L, pair_emb_dim]

        Returns:
            src: [B, L, D] updated features
            attn_weights: [B, L, L] averaged over heads
            pair_emb_updated: [B, L, pair_emb_dim] or None
            pair_mask_logit: [B, L, L] raw logit (pair_emb @ pair_emb^T)
        """
        src2, attn_weights, pair_mask_logit = self._mha_with_pair_mask(
            src, src, src, key_padding_mask=src_key_padding_mask, pair_emb=pair_emb
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Update pair embedding with residual connection
        pair_emb_updated = None
        if pair_emb is not None:
            pair_emb_delta = self.pair_emb_update_mlp(src)  # [B, L, pair_emb_dim]
            pair_emb_updated = pair_emb + pair_emb_delta

        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, attn_weights, pair_emb_updated, pair_mask_logit

    def _mha_with_pair_mask(self, query, key, value, key_padding_mask=None, pair_emb=None):
        """Manual multi-head attention with pair embedding masking.

        All inputs are [B, L, D] (batch_first=True).
        """
        B, L, D = query.shape

        # [B, L, D] -> [B, L, nhead, head_dim] -> [B, nhead, L, head_dim]
        q = self.q_proj(query).view(B, L, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).view(B, L, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).view(B, L, self.nhead, self.head_dim).permute(0, 2, 1, 3)

        # QK^T / sqrt(d_k): [B, nhead, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Pair embedding mask
        pair_mask_logit = None
        if pair_emb is not None:
            # [B, L, pair_emb_dim] @ [B, pair_emb_dim, L] = [B, L, L]
            pair_mask_logit = torch.bmm(pair_emb, pair_emb.transpose(1, 2))

            if self.pam:
                pair_mask_sigmoid = torch.sigmoid(pair_mask_logit)  # [B, L, L]
                # Expand for all heads: [B, 1, L, L] -> [B, nhead, L, L]
                pair_mask_expanded = pair_mask_sigmoid.unsqueeze(1).expand(-1, self.nhead, -1, -1)
                # Hadamard product
                attn_scores = attn_scores * pair_mask_expanded

        # Apply key padding mask
        if key_padding_mask is not None:
            # [B, L] -> [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, L, head_dim]

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # [B, L, nhead, head_dim]
        attn_output = attn_output.view(B, L, D)
        attn_output = self.out_proj(attn_output)

        # Average attention weights over heads
        attn_weights_avg = attn_weights.mean(dim=1)  # [B, L, L]

        return attn_output, attn_weights_avg, pair_mask_logit


class PAMEncoder(nn.Module):
    """Multi-layer PAM Encoder. Returns only last layer's pair_mask_logit (memory saving)."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_key_padding_mask=None, pair_emb=None):
        output = src
        current_pair_emb = pair_emb
        last_pair_mask_logit = None

        for layer in self.layers:
            output, _, pair_emb_updated, pair_mask_logit = layer(
                output, src_key_padding_mask, current_pair_emb
            )
            if pair_emb_updated is not None:
                current_pair_emb = pair_emb_updated
            if pair_mask_logit is not None:
                last_pair_mask_logit = pair_mask_logit

        return output, current_pair_emb, last_pair_mask_logit


class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts.
    Adapted from TRKT/PLA sttran.py ObjectClassifier.
    """

    def __init__(self, mode='sgdet', obj_classes=None, is_wks=True, obj_dim=192):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode
        self.is_wks = is_wks

        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img = 64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B',
                                       wv_dir='/SSD1/minseok/WS-DSGG/TRKT/PLA/lib/CSA/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = obj_dim
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).cuda(0)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        return entry

    def forward(self, entry):
        if self.mode == 'predcls':
            entry['pred_labels'] = entry['labels']
            return entry
        elif self.mode == 'sgcls':
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if self.training:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:
                entry['distribution'] = self.decoder_lin(obj_features)

                box_idx = entry['boxes'][:,0].long()
                b = int(box_idx[-1] + 1)

                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0])
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] == duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class
                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:
                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])

                im_idx = []
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]:
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = []
                for frame_id in range(len(entry['frame_names'])):
                    union_boxes_in_frame_i = union_boxes[union_boxes[:,0] == frame_id]
                    if len(union_boxes_in_frame_i) > 0:
                        union_feat.append(extract_feature_given_bbox_base_feat_torch(entry['faset_rcnn_model'], entry['transforms'], entry['cv2_imgs'][frame_id], union_boxes_in_frame_i[:, 1:], entry['fmaps'][frame_id], False))
                    else:
                        pass
                union_feat = torch.cat(union_feat)
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        else:
            # sgdet mode
            if self.is_wks or self.training:
                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)

                box_idx = entry['boxes'][:, 0][entry['pair_idx'].unique()]
                l = torch.sum(box_idx == torch.mode(box_idx)[0])
                b = int(box_idx[-1] + 1)

                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
                entry['pred_scores'] = entry['scores']
            else:
                obj_embed = entry['distribution'] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
                obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)

                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                for i in range(b):
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]

                    if scores.shape[0] != 0 and scores.shape[1] != 0:
                        for j in range(len(self.classes) - 1):
                            inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                            if inds.numel() > 0:
                                cls_dists = scores[inds]
                                cls_feats = feats[inds]
                                cls_scores = cls_dists[:, j]
                                _, order = torch.sort(cls_scores, 0, True)
                                cls_boxes = pred_boxes[inds]
                                cls_dists = cls_dists[order]
                                cls_feats = cls_feats[order]
                                keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)

                                final_dists.append(cls_dists[keep.view(-1).long()])
                                final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                            1).cuda(0),
                                                            cls_boxes[order, :][keep.view(-1).long()]), 1))
                                final_feats.append(cls_feats[keep.view(-1).long()])

                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                entry['distribution'] = torch.cat(final_dists, dim=0)
                entry['features'] = torch.cat(final_feats, dim=0)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    if entry['distribution'][box_idx == i, 0].shape[0] != 0:
                        local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0])
                        HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][entry['pred_labels'][box_idx == j] != 1]:
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX

                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)

            return entry


class DSGDETR(nn.Module):
    """
    DSG-DETR: Spatial-Temporal Transformer with DSG-DETR architecture and PAM integration.

    Key differences from TRKT/PLA STTran (transformer_img mode):
    - Spatial: 1L PAMEncoder, frame-based grouping (pad_sequence, batch_first=True)
    - Temporal: 3L PAMEncoder, object-class-based grouping + positional encoding
    - PAM: pair_emb -> attention masking (controlled by `pam` flag)
    """

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None,
                 obj_classes=None, enc_layer_num=1, dec_layer_num=3,
                 is_wks=True, feat_dim=2048, obj_dim=192, conf=None):
        super(DSGDETR, self).__init__()
        self.obj_classes = obj_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.is_wks = is_wks

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes,
                                                   is_wks=self.is_wks, obj_dim=obj_dim)

        ###################################
        self.union_func1 = nn.Conv2d(feat_dim, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256 // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.subj_fc = nn.Linear(obj_dim, 512)
        self.obj_fc = nn.Linear(obj_dim, 512)
        self.vr_fc = nn.Linear(256 * 7 * 7, 512)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B',
                                       wv_dir='/SSD1/minseok/WS-DSGG/TRKT/PLA/lib/CSA/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        d_model = 1936
        pam = getattr(conf, 'pam', True)

        # Pair affinity MLP and compressor
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
        )
        self.r_compress = nn.Linear(128, 1)

        # Positional encoding for temporal (DSG-DETR: object-class-based grouping)
        self.positional_encoder = PositionalEncoding(d_model, max_len=400)

        # Spatial encoder (frame grouping) - PAM-aware
        local_layer = PAMEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048,
                                       dropout=0.1, pair_emb_dim=128, pam=pam)
        self.local_transformer = PAMEncoder(local_layer, num_layers=enc_layer_num)

        # Temporal encoder (object class grouping) - PAM-aware
        global_layer = PAMEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048,
                                        dropout=0.1, pair_emb_dim=128, pam=pam)
        self.global_transformer = PAMEncoder(global_layer, num_layers=dec_layer_num)

        self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)

    def forward(self, entry):
        entry = self.object_classifier(entry)

        # === Visual features ===
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)  # [N, 1536]

        # === Semantic features ===
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)  # [N, 400]

        rel_features = torch.cat((x_visual, x_semantic), dim=1)  # [N, 1936]

        # === Pair embeddings ===
        pair_emb = self.pair_mlp(rel_features)  # [N, 128]

        # ============================================================
        # Spatial message passing (frame-based grouping)
        # ============================================================
        frames = []
        im_indices = entry["boxes"][entry["pair_idx"][:, 1], 0]
        for l in im_indices.unique():
            frames.append(torch.where(im_indices == l)[0])

        # Pad features and pair_emb by frame (batch_first=True)
        frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
        frame_pair_emb = pad_sequence([pair_emb[index] for index in frames], batch_first=True)
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool().to(rel_features.device)

        # Run spatial transformer
        spatial_output, spatial_pair_emb, spatial_pair_mask_logit = self.local_transformer(
            frame_features, src_key_padding_mask=masks, pair_emb=frame_pair_emb
        )

        # Unpad: concat frame outputs back to flat [N, D]
        rel_features = torch.cat([spatial_output[i, :len(index)] for i, index in enumerate(frames)])
        pair_emb = torch.cat([spatial_pair_emb[i, :len(index)] for i, index in enumerate(frames)])

        # ============================================================
        # Temporal message passing (object-class-based grouping)
        # ============================================================
        sequences = []
        for l in obj_class.unique():
            k = torch.where(obj_class.view(-1) == l)[0]
            if len(k) > 0:
                sequences.append(k)

        # Build positional indices (frame position within each sequence)
        pos_index = []
        for index in sequences:
            im_idx, counts = torch.unique(entry["pair_idx"][index][:, 0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            pos_index.append(pos)

        # Pad sequences (batch_first=True)
        sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
        seq_pair_emb = pad_sequence([pair_emb[index] for index in sequences], batch_first=True)
        masks_seq = (1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool().to(rel_features.device)
        pos_index_padded = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None

        # Apply positional encoding
        sequence_features_pe = self.positional_encoder(sequence_features, pos_index_padded)

        # Run temporal transformer
        temporal_output, temporal_pair_emb, temporal_pair_mask_logit = self.global_transformer(
            sequence_features_pe, src_key_padding_mask=masks_seq, pair_emb=seq_pair_emb
        )

        # Scatter back to original order
        rel_flat = torch.cat([temporal_output[i, :len(index)] for i, index in enumerate(sequences)])
        pair_emb_flat = torch.cat([temporal_pair_emb[i, :len(index)] for i, index in enumerate(sequences)])
        indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
        pair_indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, pair_emb.shape[1])
        assert len(indices_flat) == len(entry["pair_idx"])

        global_output = torch.zeros_like(rel_features).to(rel_features.device)
        global_output.scatter_(0, indices_flat, rel_flat)

        pair_emb_final = torch.zeros_like(pair_emb).to(pair_emb.device)
        pair_emb_final.scatter_(0, pair_indices_flat, pair_emb_flat)

        # ============================================================
        # Output
        # ============================================================
        entry["attention_distribution"] = self.a_rel_compress(global_output)
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["pair_affinity"] = self.r_compress(pair_emb_final)
        entry["rel_emb"] = pair_emb_final

        # Collect pair mask logits (last layer only, from temporal transformer)
        all_pair_mask_logits = []
        if temporal_pair_mask_logit is not None:
            all_pair_mask_logits.append(temporal_pair_mask_logit)
        elif spatial_pair_mask_logit is not None:
            all_pair_mask_logits.append(spatial_pair_mask_logit)
        entry["all_pair_mask_logits"] = all_pair_mask_logits

        # Store sequences for PAM loss mask building in training
        entry["sequences"] = sequences

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        return entry
