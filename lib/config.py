from __future__ import division
from __future__ import print_function
from os import sched_getscheduler

import numpy as np
from easydict import EasyDict as edict

__C = edict()
conf = __C

__C.gpu_id = 1
__C.multi_gpus = False
__C.num_workers = 4
__C.mode = 'sgdet'      # ['sgdet', 'sgcls', 'predcls']
__C.transformer_mode = 'org'

# training options
__C.optimizer = 'adamw' # adamw/adam/sgd
__C.lr = 1e-5
__C.nepoch = 10
__C.stop_epoch = None  # actually stop training at this epoch (scheduler still uses nepoch)
__C.enc_layer = 1
__C.dec_layer = 3
__C.is_wks = True       # weakly-supervised
__C.bce_loss = True
__C.feat_dim = 2048
__C.obj_dim = 192
__C.pseudo_way = 0      # 0代表找不到人直接扔掉，1代表找不到人随机选一个表示人
__C.remove_one_frame_video = True
__C.union_box_feature = True
__C.loss = 'BCE'        # BCE/KL/L1/L2
# knowledge distillation options
__C.teacher_mode_cfg = None
__C.temperature = None
__C.alpha = None                # soft target的权重alpha，hard target权重1-alpha
# transition module options
__C.transition_module = False
__C.t_lr = 1e-5                     # transition_module的lr
__C.IOUmatch = False                # transition时，根据IOU和label共同判断是不是相同的bbox
# label fusion options
__C.label_fusion_strategy = 0       # 0：fixed；1：adapted
# match method options
__C.det_path = '/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/AG_detection_results_refine/'
__C.feat_path = '/SSD1/minseok/WS-DSGG/TRKT/data/action-genome/AG_detection_results_refine/'
__C.lowres_factor = 1     # 1 = native; >1 = downsample images & union bbox by this factor in convert_data
__C.match = 'ori'     # ['ori, 'gtmatch', 'gdinomatch', 'exact']
__C.start_eval_epoch = 0
__C.reliability_threshold = 0.1
__C.match_threshold = 0.2
# Pair Affinity Learning and Scoring
__C.unmatched_sampling = False  # include all unmatched objects in train batch
__C.pa_loss = None    # 'bce' / 'balanced_bce' / 'focal' / 'hnm' / 'dist_bce' / 'balanced_dist_bce' / 'focal_dist' / 'hnm_dist' / 'margin_ranking' / 'soft_margin_ranking' / 'logit_bce' / None
__C.pam_loss = None   # 'triplet' / 'soft' / 'adaptive' / None
__C.pa_weight = 0.1
__C.pa_metric = False
__C.pa_pos_target = 1.0  # label smoothing for bce: soft target for positive
__C.pa_neg_target = 0.0  # label smoothing for bce: soft target for negative
__C.pa_margin = 1.0      # margin for margin_ranking mode (gt pairs)
__C.pam_weight = 0.3
__C.pam_margin = 1.0
__C.propagate_margin = 0.3
__C.pa_alpha_power = 3.0  # power for distance decay in dist_bce mode
__C.focal_gamma = 2.0        # focusing parameter for focal loss (0 = balanced_bce)
__C.hnm_neg_ratio = 3        # hard negative mining: neg:pos ratio
__C.pam = True  # Apply PAM (pair_emb @ pair_emb^T) to attention scores
__C.oracle_detection = False  # VidHOI: replace detector output with GT bboxes (perfect detection)
# dataset options
__C.save_path = ''
__C.model_path = ''
__C.fusion_model_path = ''
__C.data_path = ''
__C.datasize = 'large'
__C.ckpt = None
__C.ws_object_bbox_path = None # ws的训练只改这个文件

# experiment name
__C.exp_name = 'defaultExp'
__C.tensorboard_name = 'runs/scalar_example'

# credit https://github.com/tohinz/pytorch-mac-network/blob/master/code/config.py
def merge_cfg(yaml_cfg, cfg):
    if type(yaml_cfg) is not edict:
        return

    for k, v in yaml_cfg.items():
        if not k in cfg:
            raise KeyError('{} is not a valid config key'.format(k))

        old_type = type(cfg[k])
        if old_type is not type(v):
            if isinstance(cfg[k], np.ndarray):
                v = np.array(v, dtype=cfg[k].dtype)
            elif isinstance(cfg[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif cfg[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(cfg[k]),
                                                               type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                merge_cfg(yaml_cfg[k], cfg[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            cfg[k] = v



def cfg_from_file(file_name):
    import yaml
    with open(file_name, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    merge_cfg(yaml_cfg, __C)