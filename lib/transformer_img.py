import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class TransformerEncoderLayer(nn.Module):

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
            # src: [L, B, D] -> apply MLP to get delta for pair embedding
            pair_emb_delta = self.pair_emb_update_mlp(src)  # [L, B, pair_emb_dim]
            # Residual connection: pair_emb + delta
            pair_emb_updated = pair_emb + pair_emb_delta

        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights, pair_emb_updated, pair_mask_logit

    def _multihead_attention_with_pair_mask(self, query, key, value, key_padding_mask=None, pair_emb=None):
        """
        Manual multi-head attention with pair embedding masking

        Args:
            query, key, value: [L, B, D]
            key_padding_mask: [B, L]
            pair_emb: [L, B, pair_emb_dim]

        Returns:
            attn_output: [L, B, D]
            attn_weights: [B, L, L] (averaged over heads)
            pair_mask_logit: [B, L, L] (raw logit of pair_emb @ pair_emb^T, before sigmoid)
        """
        L, B, D = query.shape

        # Linear projections and reshape for multi-head
        # [L, B, D] -> [L, B, nhead, head_dim] -> [B, nhead, L, head_dim]
        q = self.q_proj(query).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(L, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        # Compute QK^T / sqrt(d_k): [B, nhead, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Generate pair embedding mask: pair_emb @ pair_emb^T
        pair_mask_logit = None
        if pair_emb is not None:
            # pair_emb: [L, B, pair_emb_dim] -> [B, L, pair_emb_dim]
            pair_emb_permuted = pair_emb.permute(1, 0, 2)  # [B, L, pair_emb_dim]

            # Compute pair mask (raw logit): [B, L, pair_emb_dim] @ [B, pair_emb_dim, L] = [B, L, L]
            pair_mask_logit = torch.bmm(pair_emb_permuted, pair_emb_permuted.transpose(1, 2))  # [B, L, L]

            if self.pam:
                # Normalize pair mask for attention masking
                pair_mask_sigmoid = torch.sigmoid(pair_mask_logit)  # [B, L, L]

                # Expand for all heads: [B, L, L] -> [B, 1, L, L] -> [B, nhead, L, L]
                pair_mask_expanded = pair_mask_sigmoid.unsqueeze(1).expand(-1, self.nhead, -1, -1)

                # Hadamard product (element-wise multiplication)
                attn_scores = attn_scores * pair_mask_expanded

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] -> [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, nhead, L, L]
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, L, head_dim]

        # Concatenate heads and project
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # [L, B, nhead, head_dim]
        attn_output = attn_output.view(L, B, D)  # [L, B, D]
        attn_output = self.out_proj(attn_output)

        # Average attention weights over heads for visualization: [B, nhead, L, L] -> [B, L, L]
        attn_weights_avg = attn_weights.mean(dim=1)

        return attn_output, attn_weights_avg, pair_mask_logit


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):

        tgt2, global_attention_weights = self.multihead2(query=global_input+position_embed, key=global_input+position_embed,
                                                         value=global_input, key_padding_mask=input_key_padding_mask)
        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask, pair_emb=None):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        current_pair_emb = pair_emb
        pair_mask_logits = []  # Collect pair mask logits from all layers

        for i, layer in enumerate(self.layers):
            output, local_attention_weights, pair_emb_updated, pair_mask_logit = layer(output, input_key_padding_mask, current_pair_emb)
            weights[i] = local_attention_weights

            # Collect pair mask logit
            if pair_mask_logit is not None:
                pair_mask_logits.append(pair_mask_logit)

            # Update pair_emb for next layer
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
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        current_pair_emb = pair_emb
        pair_mask_logits = []  # Collect pair mask logits from all layers

        for i, layer in enumerate(self.layers):
            # TransformerEncoderLayer is used as decoder layer
            output, global_attention_weights, pair_emb_updated, pair_mask_logit = layer(output, input_key_padding_mask, current_pair_emb)
            weights[i] = global_attention_weights

            # Collect pair mask logit
            if pair_mask_logit is not None:
                pair_mask_logits.append(pair_mask_logit)

            # Update pair_emb for next layer
            if pair_emb_updated is not None:
                current_pair_emb = pair_emb_updated

        if self.num_layers>0:
            return output, weights, current_pair_emb, pair_mask_logits
        else:
            return output, None, current_pair_emb, []


class transformer_img(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None, pair_emb_dim=128, pam=True):
        super(transformer_img, self).__init__()
        self.mode = mode
        self.pam = pam

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, pair_emb_dim=pair_emb_dim, pam=pam)
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)

        decoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, pair_emb_dim=pair_emb_dim, pam=pam)
        self.global_attention = TransformerEncoder(decoder_layer, dec_layer_num)

        # self.position_embedding = nn.Embedding(2, embed_dim) #present and next frame
        # nn.init.uniform_(self.position_embedding.weight)


    def forward(self, features, im_idx, pair_emb=None):
        rel_idx = torch.arange(im_idx.shape[0])

        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame
        b = int(im_idx[-1] + 1)     # frame_num+1
        rel_input = torch.zeros([l, b, features.shape[1]]).to(features.device)
        masks = torch.zeros([b, l], dtype=torch.bool).to(features.device)
        # TODO Padding/Mask maybe don't need for-loop
        for i in range(b):
            rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
            masks[i, torch.sum(im_idx == i):] = True

        retain_line = []
        for i in range(b):
            if len(torch.where(masks[i]==0)[0]) > 0:
                retain_line.append(i)
        masks = masks[retain_line]
        rel_input = rel_input[:, retain_line, :]

        # Reshape pair_emb if provided: [N, D] -> [L, B, D]
        pair_emb_reshaped = None
        if pair_emb is not None:
            pair_emb_reshaped = torch.zeros([l, b, pair_emb.shape[1]]).to(pair_emb.device)
            for i in range(b):
                num_objs = torch.sum(im_idx == i)
                if num_objs > 0:
                    pair_emb_reshaped[:num_objs, i, :] = pair_emb[im_idx == i]
            pair_emb_reshaped = pair_emb_reshaped[:, retain_line, :]

        # spatial encoder
        # local_output l*b*feature_size(same as rel_input) local_attention_weights num_layers(1)*b*l*l
        local_output, local_attention_weights, pair_emb_after_encoder, local_pair_mask_logits = self.local_attention(rel_input, masks, pair_emb_reshaped)

        # temporal decoder (global attention)
        global_output, global_attention_weights, pair_emb_final, global_pair_mask_logits = self.global_attention(local_output, masks, pair_emb_after_encoder)

        # local_output rel_num*feature_size
        global_output = (global_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]

        # Reshape final pair_emb back to [N, D] if it was updated
        pair_emb_output = None
        if pair_emb_final is not None:
            pair_emb_output = (pair_emb_final.permute(1, 0, 2)).contiguous().view(-1, pair_emb_final.shape[2])[masks.view(-1) == 0]

        # Only keep the last layer's pair mask logits to save GPU memory
        # Previously: [local_layer_1, ..., global_layer_1, ...] = multiple layers
        # Now: [global_layer_last] = len 1 (only last layer, used for emb loss)
        all_pair_mask_logits = [global_pair_mask_logits[-1]] if global_pair_mask_logits else local_pair_mask_logits[-1:]

        return global_output, global_attention_weights, local_attention_weights, pair_emb_output, all_pair_mask_logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

