import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


class TransformerEncoderLayer(nn.Module):
    """Local attention layer with pair embedding support"""

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1, pair_emb_dim=128, pam=True):
        super().__init__()
        self.pam = pam
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"

        # Manual attention implementation for pair embedding masking
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # MLP for updating pair embedding
        self.pair_emb_update_mlp = nn.Sequential(
            nn.Linear(embed_dim, pair_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(pair_emb_dim),
            nn.Linear(pair_emb_dim, pair_emb_dim)
        )

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask, pair_emb=None):
        """
        Args:
            src: [L, B, D] sequence length, batch, embedding dim
            input_key_padding_mask: [B, L]
            pair_emb: [L, B, pair_emb_dim] pair embeddings for dynamic masking

        Returns:
            src: updated features
            local_attention_weights: attention weights
            pair_emb_updated: updated pair embeddings
            pair_mask_logit: [B, L, L] raw logit (pair_emb @ pair_emb^T before sigmoid)
        """
        # Self-attention with pair embedding masking
        src2, local_attention_weights, pair_mask_logit = self._multihead_attention_with_pair_mask(
            src, src, src,
            key_padding_mask=input_key_padding_mask,
            pair_emb=pair_emb
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Update pair embedding using updated query with residual connection
        pair_emb_updated = None
        if pair_emb is not None:
            pair_emb_delta = self.pair_emb_update_mlp(src)  # [L, B, pair_emb_dim]
            pair_emb_updated = pair_emb + pair_emb_delta

        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights, pair_emb_updated, pair_mask_logit

    def _multihead_attention_with_pair_mask(self, query, key, value, key_padding_mask=None, pair_emb=None):
        """
        Manual multi-head attention with pair embedding masking
        """
        L, B, D = query.shape

        q = self.q_proj(query).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        pair_mask_logit = None
        if pair_emb is not None:
            pair_emb_permuted = pair_emb.permute(1, 0, 2)  # [B, L, pair_emb_dim]
            pair_mask_logit = torch.bmm(pair_emb_permuted, pair_emb_permuted.transpose(1, 2))
            if self.pam:
                pair_mask_sigmoid = torch.sigmoid(pair_mask_logit)
                pair_mask_expanded = pair_mask_sigmoid.unsqueeze(1).expand(-1, self.nhead, -1, -1)
                attn_scores = attn_scores * pair_mask_expanded

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(L, B, D)
        attn_output = self.out_proj(attn_output)

        attn_weights_avg = attn_weights.mean(dim=1)

        return attn_output, attn_weights_avg, pair_mask_logit


class TransformerDecoderLayer(nn.Module):
    """Global attention layer with pair embedding support for sliding window"""

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1, pair_emb_dim=128, pam=True):
        super().__init__()
        self.pam = pam
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"

        # Manual attention implementation for pair embedding masking
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # MLP for updating pair embedding
        self.pair_emb_update_mlp = nn.Sequential(
            nn.Linear(embed_dim, pair_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(pair_emb_dim),
            nn.Linear(pair_emb_dim, pair_emb_dim)
        )

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed, pair_emb=None):
        """
        Args:
            global_input: [L*2, B-1, D] - sliding window input (2 frames combined)
            input_key_padding_mask: [B-1, L*2]
            position_embed: [L*2, B-1, D]
            pair_emb: [L*2, B-1, pair_emb_dim] pair embeddings for dynamic masking

        Returns:
            tgt: updated features
            global_attention_weights: attention weights
            pair_emb_updated: updated pair embeddings
            pair_mask_logit: [B-1, L*2, L*2] raw logit
        """
        tgt2, global_attention_weights, pair_mask_logit = self._multihead_attention_with_pair_mask(
            query=global_input + position_embed,
            key=global_input + position_embed,
            value=global_input,
            key_padding_mask=input_key_padding_mask,
            pair_emb=pair_emb
        )

        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)

        # Update pair embedding
        pair_emb_updated = None
        if pair_emb is not None:
            pair_emb_delta = self.pair_emb_update_mlp(tgt)
            pair_emb_updated = pair_emb + pair_emb_delta

        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm4(tgt)

        return tgt, global_attention_weights, pair_emb_updated, pair_mask_logit

    def _multihead_attention_with_pair_mask(self, query, key, value, key_padding_mask=None, pair_emb=None):
        """Manual multi-head attention with pair embedding masking for global attention"""
        L, B, D = query.shape

        q = self.q_proj(query).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        pair_mask_logit = None
        if pair_emb is not None:
            pair_emb_permuted = pair_emb.permute(1, 0, 2)  # [B, L, pair_emb_dim]
            pair_mask_logit = torch.bmm(pair_emb_permuted, pair_emb_permuted.transpose(1, 2))
            if self.pam:
                pair_mask_sigmoid = torch.sigmoid(pair_mask_logit)
                pair_mask_expanded = pair_mask_sigmoid.unsqueeze(1).expand(-1, self.nhead, -1, -1)
                attn_scores = attn_scores * pair_mask_expanded

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(L, B, D)
        attn_output = self.out_proj(attn_output)

        attn_weights_avg = attn_weights.mean(dim=1)

        return attn_output, attn_weights_avg, pair_mask_logit


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask, pair_emb=None):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]], dtype=output.dtype, device=output.device)

        current_pair_emb = pair_emb
        pair_mask_logits = []

        for i, layer in enumerate(self.layers):
            output, local_attention_weights, pair_emb_updated, pair_mask_logit = layer(output, input_key_padding_mask, current_pair_emb)
            weights[i] = local_attention_weights

            if pair_mask_logit is not None:
                pair_mask_logits.append(pair_mask_logit)

            if pair_emb_updated is not None:
                current_pair_emb = pair_emb_updated

        if self.num_layers > 0:
            return output, weights, current_pair_emb, pair_mask_logits
        else:
            return output, None, current_pair_emb, []


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, global_input, input_key_padding_mask, position_embed, pair_emb=None):
        output = global_input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]], dtype=output.dtype, device=output.device)

        current_pair_emb = pair_emb
        pair_mask_logits = []

        for i, layer in enumerate(self.layers):
            output, global_attention_weights, pair_emb_updated, pair_mask_logit = layer(
                output, input_key_padding_mask, position_embed, current_pair_emb
            )
            weights[i] = global_attention_weights

            if pair_mask_logit is not None:
                pair_mask_logits.append(pair_mask_logit)

            if pair_emb_updated is not None:
                current_pair_emb = pair_emb_updated

        if self.num_layers > 0:
            return output, weights, current_pair_emb, pair_mask_logits
        else:
            return output, None, current_pair_emb, []


class transformer_wk(nn.Module):
    ''' Spatial Temporal Transformer with pair embedding support
        local_attention: spatial encoder (intra-frame)
        global_attention: temporal decoder (inter-frame with sliding window)
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None, pair_emb_dim=128, pam=True):
        super(transformer_wk, self).__init__()
        self.mode = mode
        self.pair_emb_dim = pair_emb_dim
        self.pam = pam

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, pair_emb_dim=pair_emb_dim, pam=pam)
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, pair_emb_dim=pair_emb_dim, pam=pam)

        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(2, embed_dim)  # present and next frame
        nn.init.uniform_(self.position_embedding.weight)

    def forward(self, features, im_idx, pair_emb=None):
        """
        Args:
            features: [N, D] - all object features
            im_idx: [N] - frame index for each object
            pair_emb: [N, pair_emb_dim] - pair embeddings for each object

        Returns:
            output: [N, D] - updated features
            global_attention_weights: attention weights from temporal decoder
            local_attention_weights: attention weights from spatial encoder
            pair_emb_output: [N, pair_emb_dim] - updated pair embeddings
            all_pair_mask_logits: list of pair mask logits [local_1, global_1, global_2, global_3]
        """
        rel_idx = torch.arange(im_idx.shape[0]).to(im_idx.device)

        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # max box number in a single frame
        b = int(im_idx[-1] + 1)  # frame_num
        rel_input = torch.zeros([l, b, features.shape[1]], dtype=features.dtype, device=features.device)
        masks = torch.zeros([b, l], dtype=torch.bool, device=features.device)

        for i in range(b):
            rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
            masks[i, torch.sum(im_idx == i):] = True

        # Remove empty frames
        retain_line = []
        for i in range(b):
            if len(torch.where(masks[i] == 0)[0]) > 0:
                retain_line.append(i)
        masks = masks[retain_line]
        rel_input = rel_input[:, retain_line, :]

        # Reshape pair_emb for local attention: [N, D] -> [L, B, D]
        pair_emb_local = None
        if pair_emb is not None:
            pair_emb_local = torch.zeros([l, b, pair_emb.shape[1]], dtype=pair_emb.dtype, device=pair_emb.device)
            for i in range(b):
                num_objs = torch.sum(im_idx == i)
                if num_objs > 0:
                    pair_emb_local[:num_objs, i, :] = pair_emb[im_idx == i]
            pair_emb_local = pair_emb_local[:, retain_line, :]

        # Spatial encoder (local attention)
        local_output, local_attention_weights, pair_emb_after_local, local_pair_mask_logits = self.local_attention(
            rel_input, masks, pair_emb_local
        )

        # Restore local_pair_mask_logits to full size [B, L, L]
        local_pair_mask_logits_full = []
        for layer_logit in local_pair_mask_logits:
            # layer_logit shape: [B', L, L] where B' is len(retain_line)
            full_logit = torch.zeros([b, l, l], dtype=layer_logit.dtype, device=features.device)
            full_logit[retain_line, :, :] = layer_logit
            local_pair_mask_logits_full.append(full_logit)
        local_pair_mask_logits = local_pair_mask_logits_full

        # Flatten local_output for global input construction
        local_output_flat = (local_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]

        # Flatten pair_emb_after_local for global input construction
        pair_emb_after_local_flat = None
        if pair_emb_after_local is not None:
            pair_emb_after_local_flat = (pair_emb_after_local.permute(1, 0, 2)).contiguous().view(-1, pair_emb_after_local.shape[2])[masks.view(-1) == 0]

        # Prepare global attention input (sliding window of 2 frames)
        global_input = torch.zeros([l * 2, b - 1, features.shape[1]], dtype=local_output_flat.dtype, device=features.device)
        position_embed = torch.zeros([l * 2, b - 1, features.shape[1]], dtype=local_output_flat.dtype, device=features.device)
        idx = -torch.ones([l * 2, b - 1]).to(features.device)
        idx_plus = -torch.ones([l * 2, b - 1], dtype=torch.long).to(features.device)

        # Prepare pair_emb for global attention (sliding window)
        pair_emb_global = None
        if pair_emb_after_local_flat is not None:
            pair_emb_global = torch.zeros([l * 2, b - 1, self.pair_emb_dim], dtype=pair_emb_after_local_flat.dtype, device=pair_emb.device)

        # Sliding window size = 2
        for j in range(b - 1):
            # Features from frame j and j+1
            frame_mask = (im_idx == j) | (im_idx == j + 1)
            num_objs = torch.sum(frame_mask)

            global_input[:num_objs, j, :] = local_output_flat[frame_mask]
            idx[:num_objs, j] = im_idx[frame_mask].float()
            idx_plus[:num_objs, j] = rel_idx[frame_mask]

            # Position embedding
            num_frame_j = torch.sum(im_idx == j)
            num_frame_j1 = torch.sum(im_idx == j + 1)
            position_embed[:num_frame_j, j, :] = self.position_embedding.weight[0]
            position_embed[num_frame_j:num_frame_j + num_frame_j1, j, :] = self.position_embedding.weight[1]

            # Pair embedding for global attention
            if pair_emb_global is not None:
                pair_emb_global[:num_objs, j, :] = pair_emb_after_local_flat[frame_mask]

        global_masks = (torch.sum(global_input.view(-1, features.shape[1]), dim=1) == 0).view(l * 2, b - 1).permute(1, 0)

        # Remove empty sliding windows
        retain_line_global = []
        for i in range(b - 1):
            if i in retain_line or i + 1 in retain_line:
                retain_line_global.append(i)

        global_masks = global_masks[retain_line_global]
        global_input = global_input[:, retain_line_global, :]
        position_embed = position_embed[:, retain_line_global, :]
        if pair_emb_global is not None:
            pair_emb_global = pair_emb_global[:, retain_line_global, :]

        # Handle edge case: no valid global windows
        if global_input.shape[1] == 0:
            # Return with consistent output format (local only, last layer only)
            all_pair_mask_logits = local_pair_mask_logits[-1:]  # only last layer
            return local_output_flat, None, local_attention_weights, pair_emb_after_local_flat, all_pair_mask_logits

        # Temporal decoder (global attention)
        global_output_1, global_attention_weights, pair_emb_after_global, global_pair_mask_logits = self.global_attention(
            global_input, global_masks, position_embed, pair_emb_global
        )

        output = torch.zeros(features.shape, dtype=global_output_1.dtype, device=features.device)

        # Restore global_output size
        global_output = torch.zeros([l * 2, b - 1, features.shape[1]], dtype=global_output_1.dtype, device=features.device)
        global_output[:, retain_line_global, :] = global_output_1

        # Restore pair_emb from global attention
        pair_emb_global_full = None
        if pair_emb_after_global is not None:
            pair_emb_global_full = torch.zeros([l * 2, b - 1, self.pair_emb_dim], dtype=pair_emb_after_global.dtype, device=pair_emb.device)
            pair_emb_global_full[:, retain_line_global, :] = pair_emb_after_global

        # Restore global_pair_mask_logits to full size [B-1, 2*L, 2*L]
        global_pair_mask_logits_full = []
        for layer_logit in global_pair_mask_logits:
            # layer_logit shape: [B-1', 2*L, 2*L] where B-1' is after retain_line_global filtering
            full_logit = torch.zeros([b - 1, l * 2, l * 2], dtype=layer_logit.dtype, device=features.device)
            full_logit[retain_line_global, :, :] = layer_logit
            global_pair_mask_logits_full.append(full_logit)
        global_pair_mask_logits = global_pair_mask_logits_full

        # Aggregate output based on mode
        pair_emb_output = torch.zeros(pair_emb.shape, dtype=pair_emb_global_full.dtype, device=pair_emb.device) if (pair_emb is not None and pair_emb_global_full is not None) else None

        if self.mode == 'both':
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]
                    if pair_emb_output is not None and pair_emb_global_full is not None:
                        mask_j = idx[:, j] == j
                        if mask_j.sum() > 0:
                            pair_emb_output[im_idx == j] = pair_emb_global_full[:, j][mask_j]

                if j == b - 2:
                    output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]
                    if pair_emb_output is not None and pair_emb_global_full is not None:
                        mask_j1 = idx[:, j] == j + 1
                        if mask_j1.sum() > 0:
                            pair_emb_output[im_idx == j + 1] = pair_emb_global_full[:, j][mask_j1]
                else:
                    output[im_idx == j + 1] = (global_output[:, j][idx[:, j] == j + 1] +
                                               global_output[:, j + 1][idx[:, j + 1] == j + 1]) / 2
                    if pair_emb_output is not None and pair_emb_global_full is not None:
                        mask_curr = idx[:, j] == j + 1
                        mask_next = idx[:, j + 1] == j + 1
                        if mask_curr.sum() > 0 and mask_next.sum() > 0:
                            pair_emb_output[im_idx == j + 1] = (pair_emb_global_full[:, j][mask_curr] +
                                                                pair_emb_global_full[:, j + 1][mask_next]) / 2

        elif self.mode == 'latter':
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]
                    if pair_emb_output is not None and pair_emb_global_full is not None:
                        mask_j = idx[:, j] == j
                        if mask_j.sum() > 0:
                            pair_emb_output[im_idx == j] = pair_emb_global_full[:, j][mask_j]

                output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]
                if pair_emb_output is not None and pair_emb_global_full is not None:
                    mask_j1 = idx[:, j] == j + 1
                    if mask_j1.sum() > 0:
                        pair_emb_output[im_idx == j + 1] = pair_emb_global_full[:, j][mask_j1]

        # Only keep the last layer's pair mask logits to save GPU memory
        # Previously: [local_layer_1, global_layer_1, global_layer_2, global_layer_3] = len 4
        # Now: [global_layer_3] = len 1 (only last layer, used for emb loss)
        all_pair_mask_logits = [global_pair_mask_logits[-1]] if global_pair_mask_logits else local_pair_mask_logits[-1:]

        return output, global_attention_weights, local_attention_weights, pair_emb_output, all_pair_mask_logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
