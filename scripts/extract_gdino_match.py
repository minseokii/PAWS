from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import matplotlib.pyplot as plt
# Action Genome dataset"
import os
import pickle
import sys
# Run from PAWS repo root (PYTHONPATH already includes it)
from dataloader.action_genome import AG, cuda_collate_fn
from pdb import set_trace
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm
# GDINO attention map extraction
def extract_gdino_t2v_attention(
    model,
    inputs,
    processor=None,
    average_heads=True,
    average_layers=False,
    device=None,
):
    """
    GroundingDINO encoder fusion layer에서
    text-to-vision attention을 추출하고
    multi-scale로 분리해서 반환.

    Returns
    -------
    attn_lst : list of torch.Tensor
        [attn_t2v_p1, attn_t2v_p2, attn_t2v_p3, attn_t2v_p4]
        각 tensor shape: (num_layers, num_tokens, h*w)
    hw_lst : list of tuple
        [(h1, w1), (h2, w2), (h3, w3), (h4, w4)]
    tokens : list[str] (optional)
        tokenizer가 주어졌을 경우, input token 문자열
    """

    if device is not None:
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

    attn_cache = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            attn_cache[layer_idx] = output
        return hook

    # --- register hooks ---
    handles = []
    for i in range(len(model.model.encoder.layers)):
        m = model.model.encoder.layers[i].fusion_layer.attn
        handles.append(m.register_forward_hook(make_hook(i)))

    # --- forward ---
    with torch.no_grad():
        _ = model(**inputs)

    # --- remove hooks ---
    for h in handles:
        h.remove()

    # --- collect attention ---
    attn_t2v_lst = []
    for k in sorted(attn_cache.keys()):
        x = attn_cache[k]
        # x[1][1]: text -> vision attention (softmaxed)
        attn_t2v_lst.append(x[1][1].cpu())

    # (num_layers, num_heads, num_tokens, num_patches)
    attn_t2v = torch.stack(attn_t2v_lst, dim=0)

    if average_heads:
        attn_t2v = attn_t2v.mean(dim=1)  # mean over heads

    # shape: (num_layers, num_tokens, num_patches)
    # ---------------------------------------------

    # --- compute multi-scale sizes ---
    h, w = inputs["pixel_values"].shape[2:4]

    h1 = (h + 7) // 8
    w1 = (w + 7) // 8
    p1 = h1 * w1

    h2 = (h1 + 1) // 2
    w2 = (w1 + 1) // 2
    p2 = h2 * w2

    h3 = (h2 + 1) // 2
    w3 = (w2 + 1) // 2
    p3 = h3 * w3

    h4 = (h3 + 1) // 2
    w4 = (w3 + 1) // 2
    p4 = h4 * w4

    # --- split attention by scale ---
    attn_t2v_p1 = attn_t2v[:, :, :p1]
    attn_t2v_p2 = attn_t2v[:, :, p1:p1 + p2]
    attn_t2v_p3 = attn_t2v[:, :, p1 + p2:p1 + p2 + p3]
    attn_t2v_p4 = attn_t2v[:, :, -p4:]


    attn_lst = [
        attn_t2v_p1,
        attn_t2v_p2,
        attn_t2v_p3,
        attn_t2v_p4,
    ]

    hw_lst = [
        (h1, w1),
        (h2, w2),
        (h3, w3),
        (h4, w4),
    ]

    # --- optional: token decoding ---
    tokens = None
    if processor is not None:
        tok = processor.tokenizer
        ids = inputs["input_ids"][0].tolist()
        tokens = tok.convert_ids_to_tokens(ids)

    return attn_lst, hw_lst, tokens

# matching score calculation
def calculate_box_matching_scores_soft(attn_2d, img_shape, boxes_list, target_class=None, lambd=50.0, padding=0.4, top_k=0.7):
    """
    1. Top-K Thresholding으로 노이즈 제거
    2. 내부 에너지는 100% 보존
    3. 외부(Padding) 에너지는 거리별 감쇄 보너스 적용
    """
    Ha, Wa = attn_2d.shape
    H, W = img_shape
    device = attn_2d.device
    
    # --- [Step 1] Top-K Thresholding (노이즈 제거) ---
    # 최대값의 top_k(예: 10%) 미만 신호는 0으로 처리하여 분모 오염 방지
    threshold = attn_2d.max() * top_k
    filtered_attn = torch.where(attn_2d > threshold, attn_2d, torch.zeros_like(attn_2d))
    
    # 정규화 (Min-Max)
    attn_min, attn_max = filtered_attn.min(), filtered_attn.max()
    norm_attn = (filtered_attn - attn_min) / (attn_max - attn_min + 1e-8)
    total_sum = norm_attn.sum() + 1e-8
    
    scores = torch.zeros(len(boxes_list), device=device)

    for k, d in enumerate(boxes_list):
        if target_class is not None and int(d.get('class', -1)) != target_class:
            continue
            
        x1, y1, x2, y2 = map(float, d['rect'])
        w, h = x2 - x1, y2 - y1
        
        # [확장 영역] 패딩 적용 좌표
        ax1_ex = int(((x1 - w * padding) / W) * Wa)
        ay1_ex = int(((y1 - h * padding) / H) * Ha)
        ax2_ex = int(((x2 + w * padding) / W) * Wa)
        ay2_ex = int(((y2 + h * padding) / H) * Ha)
        
        ax1_ex, ax2_ex = max(0, ax1_ex), min(Wa, ax2_ex)
        ay1_ex, ay2_ex = max(0, ay1_ex), min(Ha, ay2_ex)
        
        roi_attn = norm_attn[ay1_ex:ay2_ex, ax1_ex:ax2_ex]
        if roi_attn.numel() == 0: continue

        # --- [Step 2] 내부 1.0, 외부 감쇄 가중치 생성 ---
        rh, rw = roi_attn.shape
        y_coords = torch.linspace(y1 - h * padding, y2 + h * padding, rh, device=device)
        x_coords = torch.linspace(x1 - w * padding, x2 + w * padding, rw, device=device)
        mesh_y, mesh_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # 박스 경계로부터의 거리 계산 (내부는 0)
        dist_x = torch.max(torch.max(x1 - mesh_x, mesh_x - x2), torch.tensor(0.0, device=device))
        dist_y = torch.max(torch.max(y1 - mesh_y, mesh_y - y2), torch.tensor(0.0, device=device))
        dist = torch.sqrt(dist_x**2 + dist_y**2)

        # 감쇄 가중치 (sigma는 박스 크기에 비례하도록 설정)
        sigma = (w + h) * 0.15 
        weight_mask = torch.exp(-dist / (sigma + 1e-8))
        
        # 가중 합산
        weighted_roi_sum = (roi_attn * weight_mask).sum()

        # --- [Step 3] 최종 스코어 계산 ---
        # Concentration (점유율)
        concentration = weighted_roi_sum / total_sum
        
        # Density (밀도): 실제 박스 면적 대비 에너지
        box_area_in_attn = max(1, (x2 - x1) / W * Wa) * max(1, (y2 - y1) / H * Ha)
        density = weighted_roi_sum / box_area_in_attn
        
        # 결합 (lambd는 50정도가 적당하며, 필요시 100으로 상향)
        scores[k] = concentration * torch.sigmoid(lambd * density)

    return scores

# GDINO attention reliability calculation
def calculate_reliability_2d(attn_2d, top_k=0.3):
    """
    (H, W) 형태의 2D Attention Map을 직접 입력받아 신뢰도를 계산합니다.
    
    Args:
        attn_2d (torch.Tensor): (H, W) 크기의 Attention Map
        top_k (float): Max 값 기준 Threshold 비율 (0~1)
        
    Returns:
        float: 0~1 사이의 신뢰도 값
    """
    h, w = attn_2d.shape
    device = attn_2d.device

    # 1. Top-K Thresholding (SNR 강화 및 노이즈 제거)
    # 이미지 내의 최대 활성값 대비 top_k 이하의 낮은 신호는 0으로 처리
    threshold = attn_2d.max() * top_k
    refined_map = torch.where(attn_2d > threshold, attn_2d, torch.zeros_like(attn_2d))
    
    # 2. 확률 분포 재설정 (Sum to 1)
    total_sum = refined_map.sum() + 1e-8
    p = refined_map / total_sum
    
    # 3. 공간적 좌표 그리드 생성
    # indexing='ij'를 통해 (h, w) 순서와 일치시킴
    y_range = torch.arange(h, device=device).float()
    x_range = torch.arange(w, device=device).float()
    y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # 4. 무게중심 (Weighted Centroid) 계산
    mu_y = torch.sum(p * y_grid)
    mu_x = torch.sum(p * x_grid)
    
    # 5. 공간적 분산 (Spatial Variance) 계산
    # 무게중심으로부터 신호가 얼마나 퍼져 있는지 측정
    var_y = torch.sum(p * (y_grid - mu_y)**2)
    var_x = torch.sum(p * (x_grid - mu_x)**2)
    
    # 6. 정규화된 표준편차 거리 계산
    # 분산의 합에 루트를 씌워 실제 '거리(픽셀)' 단위로 변환
    std_dist = torch.sqrt(var_y + var_x)
    # 이미지 대각선 길이를 기준으로 정규화 (상대적 거리 확보)
    max_dist = torch.sqrt(torch.tensor(h**2 + w**2, device=device, dtype=torch.float32))
    
    # 7. 최종 신뢰도 (Exponential Decay)
    # 표준편차 거리가 0에 가까울수록(응집될수록) 1에 수렴
    # -5.0은 변별력 상수로, 응집되지 않은 노이즈의 점수를 빠르게 떨어뜨리는 역할
    reliability = torch.exp(-5.0 * (std_dist / max_dist))
    
    return reliability.item()

def normalize_tensor(tensor, mode="l2"):
    if mode == "l1":
        tensor = tensor / (tensor.sum() + 1e-12)
    elif mode == "l2":
        tensor = F.normalize(tensor, dim=-1)
    elif mode == "minmax":
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-12)
    return tensor

# Generate Target attn_2d from attn_lst
def get_normalized_attn_map(
    attn_lst,
    hw_lst,
    scale_idx=[3],
    layer_idx=[5],
    token_idx=0,
    norm_mode="minmax",
):

    # 1. 대상이 되는 Scale들 중 최대 해상도 찾기 (Target Size 결정)
    target_h, target_w = 0, 0
    for s_idx in scale_idx:
        h, w = hw_lst[s_idx]
        if h > target_h: target_h = h
        if w > target_w: target_w = w

    # 2. 모든 Scale에서 해당 Layer/Token의 Attention 값을 수집
    raw_vecs = []
    for s_idx in scale_idx:
        # [Layers, Tokens, H*W] -> [H*W] (선택된 레이어들의 평균 반영)
        v = attn_lst[s_idx][layer_idx, token_idx].float().mean(dim=0)
        raw_vecs.append(v)

    # 3. 통합 정규화 수행
    combined_raw = torch.cat(raw_vecs)
    normalized_combined = normalize_tensor(combined_raw, mode=norm_mode)

    # 4. 각 Scale별 슬라이싱 및 Interpolation (Gaussian Blur 포함)
    all_rescaled_attns = []
    current_pos = 0
    for s_idx in scale_idx:
        h, w = hw_lst[s_idx]
        length = h * w
        
        # 정규화된 데이터에서 분할
        norm_v = normalized_combined[current_pos : current_pos + length]
        current_pos += length
        
        # 2D 변환 [1, 1, H, W]
        attn_2d = norm_v.reshape(1, 1, h, w)

        # Target Size로 확대 및 블러 처리
        if (h, w) != (target_h, target_w):
            # 1. Bilinear Interpolation
            attn_2d = F.interpolate(
                attn_2d, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            
            # 2. Gaussian Blur (확대 배율에 비례한 커널 설정)
            sigma = (target_h / h) * 0.5 
            kernel_size = int(2 * int(4 * sigma + 0.5) + 1)
            attn_2d = TF.gaussian_blur(attn_2d, [kernel_size, kernel_size], [sigma, sigma])
            
        all_rescaled_attns.append(attn_2d)

    # 5. 최종 평균 계산 및 반환
    # [N, 1, Target_H, Target_W] -> [1, Target_H, Target_W]
    final_attn = torch.cat(all_rescaled_attns, dim=0).mean(dim=0)
    
    return final_attn

# GDINO Prompt Generation
def generate_relationship_prompt(obj_label, rel_label):
    """
    Contacting 및 Spatial 관계를 분석하여 문법적으로 자연스러운 프롬프트 생성
    """
    # 1. 전처리: 언더바 제거 및 소문자화
    rel = rel_label.replace('_', ' ').lower().strip()
    obj = obj_label.lower().strip()
    
    # 2. 특수 관계 처리 (부정/기타)
    if rel in ['not contacting', 'other relationship']:
        return None

    # 3. 유형별 템플릿 분기
    
    # CASE A: 'in' 관계 (사람이 물체 안에 있는 공간적 상황)
    if rel == 'in':
        return f"{obj} that the person is in."
    
    # CASE B: 수동태 관계 (사물이 주어일 때 'by'가 포함된 경우)
    elif 'by' in rel:
        # 예: blanket that is covered by the person
        return f"{obj} that is {rel} the person."
    
    # CASE C: 전치사로 끝나는 능동 접촉 관계 (on, from으로 끝나는 경우)
    # drinking from, sitting on, leaning on, lying on, writing on 등
    elif rel.endswith('on') or rel.endswith('from'):
        # 예: cup that the person is drinking from
        return f"{obj} that the person is {rel}."
    
    # CASE D: 일반적인 능동 접촉 관계 (holding, wearing, touching 등)
    elif rel in ['holding', 'wearing', 'touching', 'twisting', 'wiping', 'eating', 'carrying']:
        # 예: shirt that the person is wearing
        return f"{obj} that the person is {rel}."
    
    # CASE E: 일반 공간 위치 관계 (above, beneath, behind, in front of 등)
    else:
        # 예: table that is in front of the person
        return f"{obj} that is {rel} the person."


def generate_prompt(gt_frame, obj_classes, rel_classes):
    obj_anno_lst = gt_frame[0][1:]
    prompt_lst = []
    for i in range(len(obj_anno_lst)):
        obj_label = obj_classes[obj_anno_lst[i]['class']]
        att_rel = rel_classes[int(obj_anno_lst[i]['attention_relationship'])]
        spa_rels = [rel_classes[int(idx + 3)] for idx in obj_anno_lst[i]['spatial_relationship']]
        con_rels = [rel_classes[int(idx + 9)] for idx in obj_anno_lst[i]['contacting_relationship']]
        
        # Contatct 우선 -> Contact에서 prompt가 None일경우 Spatial prompt generate
        prompt = generate_relationship_prompt(obj_label, con_rels[0])
        if prompt is None and len(spa_rels) > 0:
            prompt = generate_relationship_prompt(obj_label, spa_rels[0])
        if prompt is not None:
            prompt_lst.append(prompt)

    return prompt_lst



# Main Code
def main():

    reliability_threshold = 0.4
    matching_threshold = 0.1


    video_dir = 'data/action-genome/frames'
    det_dir = 'data/action-genome/PLA_det_ag_class/'    # VinVL detection cache (AG class space)
    save_dir = 'data/action-genome/PLA_gdino/'           # output: same dets + reliability + match_score
    video_list = os.listdir(video_dir)
    obj_class_file = 'data/action-genome/annotations/object_classes.txt'
    rel_class_file = 'data/action-genome/annotations/relationship_classes.txt'
    weak_anno = 'data/action-genome/annotations/weak/gt_annotation_thres02_keep1.pkl'
    full_anno_path = 'data/action-genome/annotations/object_bbox_and_relationship.pkl'
    with open(weak_anno, 'rb') as f:
        gt_annotation = pickle.load(f)
    with open(full_anno_path, 'rb') as f:
        full_anno = pickle.load(f)
    print(f'Number of videos in weak annotation: {len(gt_annotation)}')

    # Note: When using TRKT_det (AG class IDs), no OI→AG mapping needed


    # Action Genome dataset load
    conf = 'configs/oneframe.yml'
    datasize = 'large'
    data_path = 'data/action-genome/'
    AG_dataset_train = AG(mode="train", datasize=datasize, data_path=data_path, ws_object_bbox_path=weak_anno, remove_one_frame_video=False,
                        filter_nonperson_box_frame=True, filter_small_box=False)
    dataloader_train = torch.utils.data.DataLoader(
        AG_dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=cuda_collate_fn,
    )

    # GDINO model load
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


    object_classes = AG_dataset_train.object_classes
    rel_classes = AG_dataset_train.relationship_classes

    for i, data in enumerate(tqdm(dataloader_train)):
        index = data[4]
        gt_frame = AG_dataset_train.gt_annotations[index]
        frame_names = AG_dataset_train.video_list[index]

        print(f"Video {i}: {len(frame_names)} frames - {frame_names}")

        # Loop through all frames in the video
        for frame_idx in range(len(frame_names)):
            frame_name = frame_names[frame_idx]
            gt_frame_single = gt_frame[frame_idx]

            # load detections and features
            dets = np.load(os.path.join(det_dir, frame_name, 'dets.npy'), allow_pickle=True)
            feat_path = os.path.join(det_dir, frame_name, 'feat.npy')
            feats = np.load(feat_path) if os.path.exists(feat_path) else None
            dets_copy = dets.copy().tolist()

            # Initialize all detections with default scores
            for d in dets_copy:
                d['reliability'] = -1.0
                d['match_score'] = -1.0

            print(f"  Frame {frame_idx}: {frame_name}")
            '''
            for j in range (1, len(gt_frame_single)):
                obj_label = gt_frame_single[j]
                obj_class = object_classes[obj_label['class']]
                obj_bbox = obj_label['bbox']
                att_rel = rel_classes[int(obj_label['attention_relationship'])]
                spa_rels = [rel_classes[int(idx + 3)] for idx in obj_label['spatial_relationship']]
                con_rels = [rel_classes[int(idx + 9)] for idx in obj_label['contacting_relationship']]
                print(f'Object {j}: Class: {obj_class}, Attention : {att_rel}, Spatial : {spa_rels}, Contact : {con_rels}')
            '''
            # grounding dino
            image = Image.open(os.path.join(video_dir, frame_name))
            prompt_lst = generate_prompt([gt_frame_single], object_classes, rel_classes)
            obj_label = gt_frame_single[1:]
            print('  Generated Prompts:', prompt_lst)
            for i, TEXT_PROMPT in enumerate(prompt_lst):
                inputs = processor(text=TEXT_PROMPT, images=image, return_tensors="pt").to(device)
                attn_lst, hw_lst, tokens = extract_gdino_t2v_attention(
                    model,
                    inputs,
                    processor=processor,
                    average_heads=True,
                    average_layers=False,
                    device=device,
                )

                # attn_lst : len=4 , (layer#, token#, hw/scale)
                # target scale : 1,2 / target layer : 5 / target token : 0[CLS]
                attn_2d = get_normalized_attn_map(
                    attn_lst=attn_lst,
                    hw_lst=hw_lst,
                    scale_idx=[1, 2],
                    layer_idx=[5],
                    token_idx=0,
                    norm_mode="minmax",
                )
                reliability = calculate_reliability_2d(attn_2d[0], top_k=0.3)
                print(f'    Prompt {i}: Reliability Score: {reliability:.4f}')

                # Calculate match score for class-matched detections
                for d in dets_copy:
                    if d.get('class', -1) == obj_label[i]['class']:
                        match_score = calculate_box_matching_scores_soft(
                            attn_2d=attn_2d[0],
                            img_shape=image.size[::-1],
                            boxes_list=[d],
                            target_class=None,
                            lambd=50,
                            top_k=0.3,
                        )
                        # Update if this is a better match (in case multiple GTs have same class)
                        if float(match_score) > d['match_score']:
                            d['match_score'] = float(match_score)
                            d['reliability'] = reliability

            print(f'  Frame {frame_idx} matching complete:')
            print(f'    Total dets: {len(dets_copy)}')

            # save detection results (all detections with reliability and match_score)
            save_path = os.path.join(save_dir, frame_name, 'dets.npy')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            result = np.array(dets_copy, dtype=object)
            np.save(save_path, result)
            print(f'    Saved to: {save_path}')

if __name__ == "__main__":
    main()
