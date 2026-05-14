"""
Loss functions and utilities for pair_affinity learning
"""

import torch
import torch.nn.functional as F


class PairAffinityMemoryBank:
    """Memory bank for storing negative sample embeddings (FIFO queue)"""

    def __init__(self, max_size=1024, emb_dim=256):
        """
        Args:
            max_size: Maximum number of embeddings to store
            emb_dim: Dimension of embeddings
        """
        self.max_size = max_size
        self.emb_dim = emb_dim
        self.bank = []  # List of tensors

    def push(self, embeddings):
        """
        Add new embeddings to the bank (FIFO queue)

        Args:
            embeddings: [N, D] tensor of embeddings
        """
        if len(embeddings) == 0:
            return

        # Detach from computation graph and move to CPU for storage
        embeddings = embeddings.detach().cpu()
        self.bank.extend(embeddings)

        # Keep only recent max_size embeddings
        if len(self.bank) > self.max_size:
            self.bank = self.bank[-self.max_size:]

    def get_all(self, device='cuda'):
        """
        Get all embeddings in the bank

        Args:
            device: Device to move tensors to

        Returns:
            [M, D] tensor of stored embeddings, or None if empty
        """
        if len(self.bank) == 0:
            return None
        return torch.stack(self.bank).to(device)

    def __len__(self):
        """Return number of embeddings in the bank"""
        return len(self.bank)

    def clear(self):
        """Clear all embeddings from the bank"""
        self.bank = []


def infonce_loss(anchor, positive, negatives, temperature=0.07):
    """
    InfoNCE (Contrastive) Loss

    Encourages anchor to be similar to positive and dissimilar to negatives.

    Args:
        anchor: [N, D] anchor embeddings
        positive: [N, D] positive embeddings (same class as anchor)
        negatives: [M, D] negative embeddings (different class)
        temperature: temperature scaling parameter (default: 0.07)

    Returns:
        scalar loss value

    Reference:
        - SimCLR: https://arxiv.org/abs/2002.05709
        - MoCo: https://arxiv.org/abs/1911.05722
    """
    # Normalize embeddings to unit sphere
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)

    # Positive similarity: [N]
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature

    # Negative similarities: [N, M]
    neg_sim = torch.matmul(anchor, negatives.T) / temperature

    # Concatenate: [N, 1+M] where positive is at index 0
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

    # Labels: positive is always at index 0
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    # Cross-entropy loss
    return F.cross_entropy(logits, labels)


def pair_mask_triplet_loss(pair_mask_logit, positive_mask, margin=1.0):
    """
    Triplet Margin Loss for Pair Mask

    Encourages positive pairs to attend to each other more than negative pairs.
    This helps the transformer focus on meaningful relationships and share context
    among related pairs while ignoring irrelevant static objects.

    For each positive anchor pair:
        - Positive: other positive pairs (with GT relationships)
        - Negative: negative pairs (without GT relationships)
        - Loss: max(0, margin - (anchor_to_pos - anchor_to_neg))

    Args:
        pair_mask_logit: [B, N, N] raw logit from pair_emb @ pair_emb^T
                         pair_mask_logit[b, i, j] = attention score from pair_i to pair_j
        positive_mask: [B, N] binary mask, 1 if pair has GT relationship, 0 if negative sample
        margin: margin value (default: 1.0)

    Returns:
        scalar loss value

    Example:
        Frame 1: pair_1 (holding cup) ✓, pair_2 (sitting chair) ✓, pair_3 (near wall) ✗
        Frame 2: pair_4 (holding phone) ✓, pair_5 (touching table) ✓, pair_6 (near window) ✗

        For anchor = pair_1:
            Positive: pair_2, pair_4, pair_5 (other positive pairs, cross-frame included)
            Negative: pair_3, pair_6 (negative pairs)

            loss += max(0, margin - (pair_mask[1,2] - pair_mask[1,3]))
            loss += max(0, margin - (pair_mask[1,2] - pair_mask[1,6]))
            loss += max(0, margin - (pair_mask[1,4] - pair_mask[1,3]))
            ...
    """
    B, N, _ = pair_mask_logit.shape
    device = pair_mask_logit.device

    total_loss = 0.0
    num_triplets = 0

    for b in range(B):
        # Get positive and negative indices
        pos_mask = positive_mask[b] == 1  # [N]
        neg_mask = positive_mask[b] == 0  # [N]

        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()

        # Skip if no positives or no negatives
        if num_pos == 0 or num_neg == 0:
            continue

        # Extract logits for positive anchors
        # anchor_logits: [N_pos, N] - logits from positive anchors to all pairs
        anchor_logits = pair_mask_logit[b, pos_mask, :]  # [N_pos, N]

        # anchor_to_pos: [N_pos, N_pos] - logits from positive anchors to positive pairs
        anchor_to_pos = anchor_logits[:, pos_mask]  # [N_pos, N_pos]

        # anchor_to_neg: [N_pos, N_neg] - logits from positive anchors to negative pairs
        anchor_to_neg = anchor_logits[:, neg_mask]  # [N_pos, N_neg]

        # Remove self-similarity (diagonal) from positive pairs
        # We don't want anchor attending to itself as a "positive"
        # Create mask to exclude diagonal: [N_pos, N_pos]
        non_diag_mask = ~torch.eye(num_pos, dtype=torch.bool, device=device)

        # Expand dimensions for broadcasting
        # anchor_to_pos: [N_pos, N_pos] -> [N_pos, N_pos, 1]
        # anchor_to_neg: [N_pos, N_neg] -> [N_pos, 1, N_neg]
        # Result: [N_pos, N_pos, N_neg] - all triplet combinations
        pos_expanded = anchor_to_pos.unsqueeze(2)  # [N_pos, N_pos, 1]
        neg_expanded = anchor_to_neg.unsqueeze(1)  # [N_pos, 1, N_neg]

        # Compute triplet loss: max(0, margin - (pos - neg))
        triplet_loss = F.relu(margin - (pos_expanded - neg_expanded))  # [N_pos, N_pos, N_neg]

        # Apply non-diagonal mask to exclude self-similarity
        # non_diag_mask: [N_pos, N_pos] -> [N_pos, N_pos, 1]
        triplet_loss = triplet_loss * non_diag_mask.unsqueeze(2).float()

        # Sum all valid triplets
        total_loss += triplet_loss.sum()
        num_triplets += non_diag_mask.sum().item() * num_neg

    # Average over all triplets
    if num_triplets > 0:
        return total_loss / num_triplets
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def pair_mask_triplet_loss_with_mask(pair_mask_logit, pos_pair_mask, margin=1.0, valid_mask=None):
    """
    Triplet Margin Loss with explicit pair-wise positive mask.

    Unlike pair_mask_triplet_loss which uses a 1D mask per batch,
    this function takes a 2D mask [B, N, N] directly specifying which
    (i, j) pairs are positive.

    This is useful for applying different margins to different pair combinations:
    - (1gt, 1gt): highest confidence -> gt_margin
    - (1, 1): lowest confidence -> propagate_margin
    - (1gt, 1): mixed -> mixed_margin

    Args:
        pair_mask_logit: [B, N, N] raw logit from pair_emb @ pair_emb^T
        pos_pair_mask: [B, N, N] boolean mask, True if (i,j) pair is positive
        margin: margin value for triplet loss
        valid_mask: [B, N, N] optional boolean mask, True if (i,j) is a valid pair (not padding)
                   If provided, only valid positions are considered for negatives.

    Returns:
        scalar loss value
    """
    B, N, _ = pair_mask_logit.shape
    device = pair_mask_logit.device

    total_loss = 0.0
    num_triplets = 0

    # Negative mask: positions that are not positive
    # If valid_mask is provided, exclude padding positions from negatives
    if valid_mask is not None:
        neg_pair_mask = ~pos_pair_mask & valid_mask  # Only valid non-positive positions
    else:
        neg_pair_mask = ~pos_pair_mask

    for b in range(B):
        pos_mask_b = pos_pair_mask[b]  # [N, N]
        neg_mask_b = neg_pair_mask[b]  # [N, N]

        # Get positive and negative (i, j) indices
        pos_indices = pos_mask_b.nonzero(as_tuple=False)  # [num_pos, 2]
        neg_indices = neg_mask_b.nonzero(as_tuple=False)  # [num_neg, 2]

        num_pos = pos_indices.shape[0]
        num_neg = neg_indices.shape[0]

        if num_pos == 0 or num_neg == 0:
            continue

        # Sample to limit memory usage (max 1000 pos * 1000 neg = 1M elements)
        max_samples = 1000
        if num_pos > max_samples:
            sample_idx = torch.randperm(num_pos, device=device)[:max_samples]
            pos_indices = pos_indices[sample_idx]
            num_pos = max_samples
        if num_neg > max_samples:
            sample_idx = torch.randperm(num_neg, device=device)[:max_samples]
            neg_indices = neg_indices[sample_idx]
            num_neg = max_samples

        # Get logit values for positive and negative pairs
        pos_logits = pair_mask_logit[b, pos_indices[:, 0], pos_indices[:, 1]]  # [num_pos]
        neg_logits = pair_mask_logit[b, neg_indices[:, 0], neg_indices[:, 1]]  # [num_neg]

        # Triplet loss: for each positive, compare with all negatives
        # pos_logits: [num_pos] -> [num_pos, 1]
        # neg_logits: [num_neg] -> [1, num_neg]
        pos_expanded = pos_logits.unsqueeze(1)  # [num_pos, 1]
        neg_expanded = neg_logits.unsqueeze(0)  # [1, num_neg]

        # loss = max(0, margin - (pos - neg))
        triplet_loss = F.relu(margin - (pos_expanded - neg_expanded))  # [num_pos, num_neg]

        total_loss += triplet_loss.sum()
        num_triplets += num_pos * num_neg

    if num_triplets > 0:
        return total_loss / num_triplets
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def pair_mask_adaptive_margin_loss(pair_mask_logit, pos_pair_mask, confidence, base_margin=1.0, valid_mask=None):
    """
    Adaptive Margin Triplet Loss.

    Teacher confidence에 비례하여 per-triplet margin을 조절.
    확신하는 pos/neg pair에는 큰 margin, 애매한 pair에는 작은 margin.

    margin_ij = base_margin * conf_pos_i * conf_neg_j
    where:
        conf_pos = (target_logit - 0.5) * 2    # 0.5→0, 1.0→1
        conf_neg = (0.5 - target_logit) * 2    # 0.5→0, 0.0→1

    Args:
        pair_mask_logit: [B, N, N] sigmoid-ed score from pair_emb @ pair_emb^T (0~1 range)
        pos_pair_mask: [B, N, N] boolean mask, True if (i,j) is positive
        confidence: [B, N] teacher's blended target_logit per pair (0~1 range)
        base_margin: maximum margin (default: 1.0)
        valid_mask: [B, N, N] optional boolean mask for valid (non-padding) positions

    Returns:
        scalar loss value
    """
    B, N, _ = pair_mask_logit.shape
    device = pair_mask_logit.device

    total_loss = 0.0
    num_triplets = 0

    # Negative mask: positions that are not positive
    if valid_mask is not None:
        neg_pair_mask = ~pos_pair_mask & valid_mask
    else:
        neg_pair_mask = ~pos_pair_mask

    for b in range(B):
        pos_mask_b = pos_pair_mask[b]  # [N, N]
        neg_mask_b = neg_pair_mask[b]  # [N, N]

        # Get positive and negative (i, j) indices
        pos_indices = pos_mask_b.nonzero(as_tuple=False)  # [num_pos, 2]
        neg_indices = neg_mask_b.nonzero(as_tuple=False)  # [num_neg, 2]

        num_pos = pos_indices.shape[0]
        num_neg = neg_indices.shape[0]

        if num_pos == 0 or num_neg == 0:
            continue

        # Sample to limit memory usage
        max_samples = 1000
        if num_pos > max_samples:
            sample_idx = torch.randperm(num_pos, device=device)[:max_samples]
            pos_indices = pos_indices[sample_idx]
            num_pos = max_samples
        if num_neg > max_samples:
            sample_idx = torch.randperm(num_neg, device=device)[:max_samples]
            neg_indices = neg_indices[sample_idx]
            num_neg = max_samples

        # Get logit values for positive and negative pairs
        pos_logits = pair_mask_logit[b, pos_indices[:, 0], pos_indices[:, 1]]  # [num_pos]
        neg_logits = pair_mask_logit[b, neg_indices[:, 0], neg_indices[:, 1]]  # [num_neg]

        # Per-pair confidence from teacher's target_logit
        # pos pair i: confidence = how far above 0.5 (higher target_logit → more confident positive)
        conf_pos = ((confidence[b, pos_indices[:, 0]] - 0.5) * 2).clamp(min=0)  # [num_pos]
        # neg pair j: confidence = how far below 0.5 (lower target_logit → more confident negative)
        conf_neg = ((0.5 - confidence[b, neg_indices[:, 0]]) * 2).clamp(min=0)  # [num_neg]

        # Per-triplet adaptive margin: [num_pos, num_neg]
        margin_tensor = base_margin * conf_pos.unsqueeze(1) * conf_neg.unsqueeze(0)

        # Triplet loss with adaptive margin
        pos_expanded = pos_logits.unsqueeze(1)  # [num_pos, 1]
        neg_expanded = neg_logits.unsqueeze(0)  # [1, num_neg]

        triplet_loss = F.relu(margin_tensor - (pos_expanded - neg_expanded))  # [num_pos, num_neg]

        total_loss += triplet_loss.sum()
        num_triplets += num_pos * num_neg

    if num_triplets > 0:
        return total_loss / num_triplets
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def soft_margin_ranking_loss(pos_scores, neg_scores):
    """
    Soft Margin Ranking Loss for logit-based pair_affinity learning.

    Unlike hard margin loss which stops learning when margin is satisfied,
    soft margin loss continues to improve separation asymptotically.

    Loss = log(1 + exp(-(pos_score - neg_score)))

    Args:
        pos_scores: [N] scores for positive pairs
        neg_scores: [M] scores for negative pairs

    Returns:
        scalar loss value
    """
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=pos_scores.device, requires_grad=True)

    # Broadcast: [N, 1] - [1, M] = [N, M]
    pos_expanded = pos_scores.unsqueeze(1)  # [N, 1]
    neg_expanded = neg_scores.unsqueeze(0)  # [1, M]

    # Soft margin: log(1 + exp(-(pos - neg)))
    diff = pos_expanded - neg_expanded  # [N, M]
    loss = F.softplus(-diff)  # softplus(x) = log(1 + exp(x))

    return loss.mean()


def pair_mask_soft_margin_loss(pair_mask_logit, positive_mask):
    """
    Soft Margin Loss for Pair Mask (embedding-based).

    Similar to pair_mask_triplet_loss but uses soft margin instead of hard margin.
    No margin hyperparameter needed - loss asymptotically approaches 0.

    Loss = log(1 + exp(-(anchor_to_pos - anchor_to_neg)))

    Args:
        pair_mask_logit: [B, N, N] raw logit from pair_emb @ pair_emb^T
        positive_mask: [B, N] binary mask, 1 if pair has GT relationship

    Returns:
        scalar loss value
    """
    B, N, _ = pair_mask_logit.shape
    device = pair_mask_logit.device

    total_loss = 0.0
    num_triplets = 0

    for b in range(B):
        pos_mask = positive_mask[b] == 1
        neg_mask = positive_mask[b] == 0

        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()

        if num_pos == 0 or num_neg == 0:
            continue

        # Extract logits for positive anchors
        anchor_logits = pair_mask_logit[b, pos_mask, :]  # [N_pos, N]
        anchor_to_pos = anchor_logits[:, pos_mask]  # [N_pos, N_pos]
        anchor_to_neg = anchor_logits[:, neg_mask]  # [N_pos, N_neg]

        # Remove self-similarity (diagonal)
        non_diag_mask = ~torch.eye(num_pos, dtype=torch.bool, device=device)

        # Expand for broadcasting: [N_pos, N_pos, N_neg]
        pos_expanded = anchor_to_pos.unsqueeze(2)  # [N_pos, N_pos, 1]
        neg_expanded = anchor_to_neg.unsqueeze(1)  # [N_pos, 1, N_neg]

        # Soft margin loss: log(1 + exp(-(pos - neg)))
        diff = pos_expanded - neg_expanded  # [N_pos, N_pos, N_neg]
        triplet_loss = F.softplus(-diff)  # [N_pos, N_pos, N_neg]

        # Apply non-diagonal mask
        triplet_loss = triplet_loss * non_diag_mask.unsqueeze(2).float()

        total_loss += triplet_loss.sum()
        num_triplets += non_diag_mask.sum().item() * num_neg

    if num_triplets > 0:
        return total_loss / num_triplets
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def distance_weighted_bce_loss(
    student_logits,
    teacher_logits,
    positive_mask,
    negative_mask,
    im_idx,
    alpha_power=3.0,
    bce_weight=1.0
):
    """
    Distance-weighted BCE loss for pair_affinity learning.

    Combines propagation pseudo labels with teacher predictions using
    distance-based weighting. Pairs closer to center frame trust propagation
    more, while pairs at edges rely more on teacher predictions.

    target = alpha * pseudo_gt + (1 - alpha) * sigmoid(teacher_logit)
    alpha = (1 - distance / max_distance) ** power

    Args:
        student_logits: [N] student model's pair_affinity logits
        teacher_logits: [N] teacher model's pair_affinity logits
        positive_mask: [N] boolean mask, True for positive (propagated) pairs
        negative_mask: [N] boolean mask, True for negative pairs
        im_idx: [N] frame index for each pair
        alpha_power: power for decay (default: 3.0)
        bce_weight: weight for the loss (default: 1.0)

    Returns:
        scalar loss value

    Example (30 frames, center=15, power=3):
        distance=0:  alpha=1.00 -> 100% propagation
        distance=5:  alpha=0.30 -> 70% teacher
        distance=10: alpha=0.04 -> 96% teacher
        distance=15: alpha=0.00 -> 100% teacher
    """
    if positive_mask.sum() == 0 or negative_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # Compute frame-based alpha
    num_frames = int(im_idx.max().item()) + 1
    center_frame = num_frames // 2
    max_distance = num_frames / 2.0

    distances = torch.abs(im_idx.float() - center_frame)
    alpha = (1.0 - distances / max_distance).clamp(min=0.0) ** alpha_power

    t_sigmoid = torch.sigmoid(teacher_logits)

    # Positive group: pseudo_gt=1
    # target = alpha * 1.0 + (1 - alpha) * t_sigmoid
    pos_alpha = alpha[positive_mask]
    pos_t_sigmoid = t_sigmoid[positive_mask]
    pos_target = pos_alpha + (1.0 - pos_alpha) * pos_t_sigmoid
    pos_bce = F.binary_cross_entropy_with_logits(
        student_logits[positive_mask], pos_target
    )

    # Negative group: pseudo_gt=0
    # target = alpha * 0.0 + (1 - alpha) * t_sigmoid = (1 - alpha) * t_sigmoid
    neg_alpha = alpha[negative_mask]
    neg_t_sigmoid = t_sigmoid[negative_mask]
    neg_target = (1.0 - neg_alpha) * neg_t_sigmoid
    neg_bce = F.binary_cross_entropy_with_logits(
        student_logits[negative_mask], neg_target
    )

    # Balanced: average of pos and neg group losses
    return bce_weight * (pos_bce + neg_bce) / 2
