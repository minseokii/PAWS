# Action Genome Dataset Dataloader 설명

## 개요

Action Genome dataset은 person-centric video scene graph generation을 위한 데이터셋입니다. 각 frame마다 1명의 person과 여러 objects 간의 관계를 annotation합니다.

## Dataset 클래스: `AG`

**위치**: `/SSD1/minseok/WS-DSGG/TRKT/PLA/dataloader/action_genome.py`

### 초기화 파라미터

```python
AG(mode, datasize, data_path='', filter_nonperson_box_frame=True,
   filter_small_box=False, ws_object_bbox_path=None, need_relation=True)
```

| 파라미터 | 설명 | 값 |
|---------|------|-----|
| `mode` | train/test 모드 | `'train'` or `'test'` |
| `datasize` | 데이터셋 크기 | `'large'` (전체), `'mini'` (1000 frames) |
| `data_path` | 데이터 경로 | `/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/` |
| `filter_nonperson_box_frame` | person bbox 없는 frame 제거 여부 | `True` (기본값) |
| `ws_object_bbox_path` | Weakly-supervised annotation 경로 | `'annotations/weak/gt_annotation_thres02_keep1.pkl'` 또는 `None` |
| `need_relation` | relation annotation 필수 여부 | `True` (train), `False` (test) |

### `__getitem__` 반환값

```python
data = dataset[index]
```

**반환**: `(img_tensor, im_info, gt_boxes, num_boxes, index)` 튜플 (총 5개 요소)

#### `data[0]`: img_tensor
- **타입**: `torch.Tensor`
- **Shape**: `[num_frames, 3, H, W]`
  - `num_frames`: 비디오의 frame 수 (최소 3개)
  - `3`: RGB 채널
  - `H, W`: resize된 height, width (target_size=600, max_size=1000)
- **dtype**: `torch.float32`
- **용도**: Faster R-CNN 입력 이미지
- **전처리**:
  - Mean subtraction: `[102.9801, 115.9465, 122.7717]`
  - Resize: 짧은 변을 600으로, 긴 변 최대 1000
  - Channel order: RGB (BGR 아님!)
  - Permuted: `[N, H, W, C]` → `[N, C, H, W]`

#### `data[1]`: im_info
- **타입**: `torch.Tensor`
- **Shape**: `[num_frames, 3]`
- **dtype**: `torch.float32`
- **내용**: 각 frame의 `[height, width, scale]`
  - `height`: resize 후 높이
  - `width`: resize 후 너비
  - `scale`: resize 비율 (resize 후 크기 / 원본 크기)
- **용도**: Detection 결과를 원본 이미지 좌표로 변환

#### `data[2]`: gt_boxes
- **타입**: `torch.Tensor`
- **Shape**: `[num_frames, 1, 5]`
- **dtype**: `torch.float32`
- **내용**: **모두 0으로 채워진 빈 tensor**
- **용도**: Weakly-supervised 설정에서는 사용하지 않음
  - Detection bbox는 refine 폴더에서 로드
  - GT bbox는 사용하지 않음 (fully-supervised가 아니므로)

#### `data[3]`: num_boxes
- **타입**: `torch.Tensor`
- **Shape**: `[num_frames]`
- **dtype**: `torch.int64`
- **내용**: **모두 0으로 채워진 빈 tensor**
- **용도**: Weakly-supervised 설정에서는 사용하지 않음

#### `data[4]`: index
- **타입**: `int`
- **내용**: 데이터셋의 비디오 인덱스 (0부터 시작)
- **용도**: GT annotation 및 frame 이름 가져오기
  ```python
  gt_annotation = AG_dataset.gt_annotations[data[4]]
  frame_names = AG_dataset.video_list[data[4]]
  ```

### 예시

```python
# Dataset 생성
dataset = AG(
    mode='train',
    datasize='large',
    data_path='/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/',
    ws_object_bbox_path='annotations/weak/gt_annotation_thres02_keep1.pkl'
)

# 데이터 로드
img_tensor, im_info, gt_boxes, num_boxes, index = dataset[0]

print(f'img_tensor.shape: {img_tensor.shape}')    # [5, 3, 600, 800] (예시)
print(f'im_info.shape: {im_info.shape}')          # [5, 3]
print(f'gt_boxes.shape: {gt_boxes.shape}')        # [5, 1, 5]
print(f'num_boxes.shape: {num_boxes.shape}')      # [5]
print(f'index: {index}')                          # 0

# GT annotation 가져오기
gt_annotation = dataset.gt_annotations[index]
frame_names = dataset.video_list[index]
print(f'Number of frames: {len(frame_names)}')    # 5
print(f'First frame: {frame_names[0]}')           # '001YG.mp4/000089.png'
```

## Dataset 속성

### 주요 속성

| 속성 | 타입 | 설명 |
|------|------|------|
| `video_list` | `List[List[str]]` | 각 비디오의 frame 이름 목록 (e.g., `'001YG.mp4/000089.png'`) |
| `gt_annotations` | `List[List[Dict]]` | WS annotation (keep1.pkl 또는 modelfree_05.pkl) |
| `real_gt_annotations` | `List[List[Dict]]` | 원본 완전한 GT (사용 안 함) |
| `video_size` | `List[Tuple[int, int]]` | 각 비디오의 (width, height) |
| `object_classes` | `List[str]` | 36개 object class 이름 |
| `relationship_classes` | `List[str]` | 26개 relationship class 이름 |

### Object Classes (36개)

```python
self.object_classes = [
    '__background__', 'person', 'bag', 'bed', 'blanket', 'book',
    'box', 'broom', 'chair', 'closet/cabinet', 'clothes',
    'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
    'floor', 'food', 'groceries', 'laptop', 'light',
    'medicine', 'mirror', 'paper/notebook', 'phone/camera', 'picture',
    'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe',
    'sofa/couch', 'table', 'television', 'towel', 'vacuum',
    'window'
]
```

### Relationship Classes (26개)

**Attention Relationships (3개)** - Single-label:
- `looking_at` (0)
- `not_looking_at` (1)
- `unsure` (2)

**Spatial Relationships (6개)** - Multi-label:
- `in_front_of` (3)
- `not_in_front_of` (4)
- `on_the_side_of` (5)
- `in` (6)
- `carrying` (7)
- `covered_by` (8)

**Contacting Relationships (17개)** - Multi-label:
- `drinking_from` (9)
- `eating` (10)
- `have_it_on_the_back` (11)
- `holding` (12)
- `leaning_on` (13)
- `lying_on` (14)
- `not_contacting` (15)
- `other_relationship` (16)
- `sitting_on` (17)
- `standing_on` (18)
- `touching` (19)
- `twisting` (20)
- `wearing` (21)
- `wiping` (22)
- `writing_on` (23)
- `opening` (24) - modelfree에만 있음
- `closing` (25) - modelfree에만 있음

## GT Annotation 구조

### `gt_annotations[index]` - Frame-level annotation list

```python
gt_annotation = dataset.gt_annotations[index]
# Type: List[List[Dict]]
# Length: num_frames in the video
```

**각 frame의 annotation**:
```python
frame_annotation = gt_annotation[frame_idx]
# Type: List[Dict]
# Length: 1 (person) + num_objects

# frame_annotation[0]: Person info
{
    'person_bbox': np.array([[x1, y1, x2, y2]]),  # Shape: [1, 4], xyxy format
}

# frame_annotation[1:]: Object info (각 object마다)
{
    'class': int,                                  # Object class index (0-35)
    'bbox': (x, y, w, h),                         # xywh format (주의: person과 다름!)
    'attention_relationship': torch.Tensor([rel_idx]),     # Shape: [1]
    'spatial_relationship': torch.Tensor([rel_idx, ...]),  # Shape: [num_spatial_rels]
    'contacting_relationship': torch.Tensor([rel_idx, ...]), # Shape: [num_contact_rels]
    'metadata': {'tag': str, 'set': 'train'/'test'},
    'visible': bool,
    'object_source': {
        'class': ['gt'],
        'bbox': ['gt'] or ['de'],
        'ar': ['gt', 're'] or ['1gt'] or ['1'],
        'sr': ['gt', 're'] or ['1gt'] or ['1'],
        'cr': ['gt', 're'] or ['1gt'] or ['1'],
        'visible': ['gt']
    }
}
```

### Object Source 의미

| Source | 의미 |
|--------|------|
| `'gt'` | Ground truth (사람이 직접 annotation) |
| `'de'` | Detection (detector가 예측) |
| `'re'` | Refined (GT를 기반으로 refine) |
| `'1gt'` | 1-frame GT에서 propagation (keep1.pkl의 1 frame GT) |
| `'1'` | Model-free temporal propagation (pseudo-label) |

## Training Pipeline에서의 사용

```python
# train.py에서의 사용 예시
for data in dataloader_train:
    img_tensor = data[0].cuda(0)      # [N, 3, H, W] - 입력 이미지
    im_info = data[1].cuda(0)         # [N, 3] - 이미지 정보
    gt_boxes = data[2].cuda(0)        # [N, 1, 5] - 빈 tensor (사용 안 함)
    num_boxes = data[3].cuda(0)       # [N] - 빈 tensor (사용 안 함)
    video_idx = data[4]               # int - 비디오 인덱스

    # GT annotation 가져오기
    gt_annotation = AG_dataset_train.gt_annotations[video_idx]
    frame_names = AG_dataset_train.video_list[video_idx]

    # Object detector 호출 (refine 폴더에서 detection 로드)
    entry = object_detector(img_tensor, im_info, gt_boxes, num_boxes,
                           gt_annotation, frame_names, ...)

    # Model forward
    pred = model(entry)
```

## 주요 차이점: Train vs Test

| 속성 | Train | Test |
|------|-------|------|
| `ws_object_bbox_path` | keep1.pkl 또는 modelfree_05.pkl | keep1.pkl |
| `need_relation` | `True` (relation 필수) | `False` (relation 없어도 됨) |
| Frame filtering | Relation 있는 frame만 | 모든 valid frame |
| Video filtering | 최소 3 frames | 최소 3 frames |
| Detection source | refine 폴더 | refine 폴더 |
| GT annotation 용도 | class + relation (bbox 사용 안 함) | bbox + class + relation (전부 사용) |

## 데이터 통계

### Train Set (keep1.pkl)
- 비디오 수: 7,649개
- 프레임 수: 167,068개
- Supervision level:
  - Weakly-supervised: 47.4%
  - Partially-supervised: 48.8%
  - Fully-supervised: 3.8%

### Test Set (keep1.pkl)
- 비디오 수: 1,776개
- 프레임 수: 54,429개
- Supervision: 100% fully-supervised

### Train Set (modelfree_05.pkl)
- 비디오 수: 6,032개 (1,617개 감소)
- 프레임 수: 91,525개
- Supervision: Model-free temporal propagation (pseudo-label)
- 감소 이유: Temporal IoU matching 실패 (threshold=0.5)

## Frame Filtering 조건

### Training (`need_relation=True`)
Frame이 valid하려면:
1. Person bbox가 존재해야 함
2. Object bbox 개수 > 0
3. 최소 1개 object가:
   - `visible == True`
   - `attention_relationship != None and != []`
   - `spatial_relationship != None and != []`
   - `contacting_relationship != None and != []`

### Testing (`need_relation=False`)
Frame이 valid하려면:
1. Person bbox가 존재해야 함
2. Object bbox 개수 > 0
3. 최소 1개 object가 `visible == True`

## Video Filtering 조건

- 최소 3개 frames 이상이어야 함 (Line 164)
- `remove_one_frame_video=True`인 경우 1-frame video 제거

## 참고사항

### Bbox Format 차이
- **Person bbox**: `xyxy` format (x1, y1, x2, y2)
- **Object bbox**: `xywh` format (x, y, width, height)

### Evaluation 시 주의사항
- Evaluation 시 GT bbox를 사용하므로 keep1.pkl 필요
- modelfree_05.pkl은 test set이 없으므로 evaluation 불가
- Detection 결과를 GT bbox와 IoU 비교 (threshold=0.5)

### Real GT Annotations
- `real_gt_annotations`는 로드되지만 **사용되지 않음**
- 코드에 남아있는 legacy code
- 원본 완전한 GT annotation (person_bbox.pkl, object_bbox_and_relationship.pkl)

## 파일 구조

```
/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/
├── frames/                          # 원본 이미지
│   └── {video_name}/{frame_name}    # e.g., 001YG.mp4/000089.png
├── annotations/
│   ├── weak/
│   │   ├── gt_annotation_thres02_keep1.pkl                      # WS annotation (train+test)
│   │   └── gt_annotation_thres02_keep1_modelfree_05.pkl        # Model-free propagation (train only)
│   ├── person_bbox.pkl              # 원본 person GT (사용 안 함)
│   └── object_bbox_and_relationship.pkl  # 원본 object GT (사용 안 함)
└── AG_detection_results_refine/     # Detection 결과
    └── {video_name}/{frame_name}/
        ├── dets.npy                 # Detection boxes
        └── feat.npy                 # Visual features
```

## 요약

**Action Genome Dataloader의 핵심**:
1. `__getitem__`은 5개 요소를 반환: 이미지, 이미지 정보, 빈 GT boxes, 빈 num_boxes, 비디오 인덱스
2. 실제 GT annotation은 `dataset.gt_annotations[index]`로 별도 접근
3. Detection bbox는 refine 폴더에서 로드 (pkl의 GT bbox는 학습 시 사용 안 함)
4. Training: class + relation만 사용, Evaluation: bbox + class + relation 모두 사용
5. Test set은 keep1.pkl 사용 (modelfree_05.pkl은 train만 있음)
