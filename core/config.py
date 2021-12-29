#! /usr/bin/env python
# coding=utf-8


from easydict import EasyDict as edict
from filters import *
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_num', dest='exp_num', type=str, default='101', help='current experiment number')
parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int, default=0, help='# of epochs')
parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int, default=80, help='# of epochs')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--exp_dir', dest='exp_dir', default='./experiments', help='models are saved here')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='7', help='if use gpu, use gpu device id')
parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use DIP Module')
parser.add_argument('--fog_FLAG', dest='fog_FLAG', type=bool, default=True, help='whether use Hybrid data training')
parser.add_argument('--vocfog_traindata_dir', dest='vocfog_traindata_dir', default='/data/vdd/liuwenyu/data_vocfog/train/JPEGImages/',
                    help='the dir contains ten levels synthetic foggy images')
parser.add_argument('--vocfog_valdata_dir', dest='vocfog_valdata_dir', default='/data/vdd/liuwenyu/data_vocfog/val/JPEGImages/',
                    help='the dir contains ten levels synthetic foggy images')
parser.add_argument('--train_path', dest='train_path', nargs='*', default='./data/dataset_fog/voc_norm_train.txt', help='folder of the training data')
parser.add_argument('--val_path', dest='val_path', nargs='*', default='./data/dataset_fog/voc_norm_test.txt', help='folder of the training data')
parser.add_argument('--test_path', dest='test_path', nargs='*', default='./data/dataset_fog/quick_test.txt', help='folder of the training data')
parser.add_argument('--class_name', dest='class_name', nargs='*', default='./data/classes/vocfog.names', help='folder of the training data')
parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', nargs='*', default='./experiments/exp_101/detection_results/', help='folder of the training data')
parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', nargs='*', default='./experiments/exp_101/checkpoint/yolov3_test_loss=5.8980.ckpt-75', help='folder of the training data')
parser.add_argument('--pre_train', dest='pre_train', default='NULL', help='the path of pretrained models if is not null. not used for now')
# we trained our model from scratch.





args = parser.parse_args()



__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

###########################################################################
# Filter Parameters
###########################################################################


cfg.filters = [
    DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter
]
cfg.num_filter_parameters = 15

cfg.defog_begin_param = 0

cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14


cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)



# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

###########################################################################
# CNN Parameters
###########################################################################
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096

###########################################################################

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = args.class_name
__C.YOLO.ANCHORS                = "./data/anchors/coco_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ISP_FLAG            = args.ISP_FLAG


# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = args.train_path
__C.TRAIN.BATCH_SIZE            = 6
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]

# __C.TRAIN.INPUT_SIZE            = [512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = args.epoch_first_stage
__C.TRAIN.SECOND_STAGE_EPOCHS   = args.epoch_second_stage
__C.TRAIN.INITIAL_WEIGHT        = args.pre_train



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = args.val_path
__C.TEST.BATCH_SIZE             = 6
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = args.WRITE_IMAGE_PATH
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = args.WEIGHT_FILE
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45






