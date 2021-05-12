#! /usr/bin/env python
# coding=utf-8


from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES          = "E:/project/zg_detection/detection/data/classes/txzg-food38.names"
__C.YOLO.ANCHORS          = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES          = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH  = 0.5
__C.YOLO.UPSAMPLE_METHOD  = "resize"
__C.YOLO.ORIGINAL_WEIGHT  = ""
__C.YOLO.DEMO_WEIGHT      = "./checkpoint/yolov3_train_loss=2.7829.ckpt-50"

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH          = "E:/project/zg_detection/detection/data/dataset/food38_train.txt"
__C.TRAIN.BATCH_SIZE          = 2
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LEARN_RATE_INIT     = 1e-4
__C.TRAIN.LEARN_RATE_END      = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS  = 10
__C.TRAIN.SECOND_STAGE_EPOCHS = 30
__C.TRAIN.INITIAL_WEIGHT      = ""

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH            = "E:/project/zg_detection/detection/data/dataset/food38_test.txt"
__C.TEST.BATCH_SIZE            = 2
__C.TEST.INPUT_SIZE            = 416
__C.TEST.DATA_AUG              = False
__C.TEST.WRITE_IMAGE           = True
__C.TEST.WRITE_IMAGE_PATH      = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_train_loss=2.7829.ckpt-50"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.8
__C.TEST.IOU_THRESHOLD          = 0.45
