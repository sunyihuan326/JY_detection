from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES                 = "E:/JY_detection/xdsj_detection/data/classes/JoyRobot_20.names"
__C.YOLO.ANCHORS                 = "./data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY        = 0.9995
__C.YOLO.STRIDES                 = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE        = 3
__C.YOLO.IOU_LOSS_THRESH         = 0.5
__C.YOLO.UPSAMPLE_METHOD         = "resize"
__C.YOLO.ORIGINAL_WEIGHT         = "F:/BaiduNetdiskDownload/tensorflow-yolov3/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT             = "./checkpoint/yolov3_test_loss=1.5417.ckpt-55"

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH           = "./data/dataset/train19_1025.txt"
__C.TRAIN.BATCH_SIZE           = 4
__C.TRAIN.INPUT_SIZE           = 416
__C.TRAIN.DATA_AUG             = True
__C.TRAIN.LEARN_RATE_INIT      = 1e-4
__C.TRAIN.LEARN_RATE_END       = 1e-6
__C.TRAIN.WARMUP_EPOCHS        = 1
__C.TRAIN.FISRT_STAGE_EPOCHS   = 75
__C.TRAIN.SECOND_STAGE_EPOCHS  = 0
# __C.TRAIN.INITIAL_WEIGHT             ="./checkpoint/padding_resize/mAP94/yolov3_test_loss=1.5903.ckpt-40"
__C.TRAIN.INITIAL_WEIGHT       = "./checkpoint/yolov3_test_loss=1.9800.ckpt-60"

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH             = "./data/dataset/test19_1025.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 416
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "E:/JY_detection/xdsj_detection/detection"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = False
__C.TEST.WEIGHT_FILE            = ""
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.5
__C.TEST.IOU_THRESHOLD          = 0.5
