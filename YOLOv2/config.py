import os
import numpy as np
'''
path and dataset parameter
配置文件
'''

DATA_PATH = '/content/drive/MyDrive/PapperReproduce/YOLOv2/data'
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')        # 存放输出文件的地方，data/pascal_voc/output
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')      # weights_dir, 路径为data/pascal_voc/weights
WEIGHTS_FILE = None   # weights file
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')

# PASCAL VOC数据集的20个类别
CLASSES = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']

FLIPPED = False

"""
model parameter
"""
IMAGE_SIZE = 416                        # 输入图片的大小，yolov1中输入图片大小为448，yolov2中输入图片大小为416
IMAGE_SCALE = 32
CELL_SIZE = IMAGE_SIZE//IMAGE_SCALE     # grid cell大小（cell_size * cell_size的大小），yolov1使用固定的7*7，yolov2使用图片大小除以32
BOXES_PER_CELL = 2                      # 每个cell负责预测两个bounding box，yolov2不需要
ANCHORS_PRE_CELL = 5                    # 每个grid cell分配5个anchors
ALPHA = 0.1                             # Leaky Relu的泄露参数
DISP_CONSOLE = False
YOLO_ANCHORS = np.array((               # 5个anchor尺寸，通过k-means聚类获得
    (0.57273, 0.677385), (1.87446, 2.06253),
                         (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)))
"""
下面这几个是论文中涉及的参数
"""
OBJECT_SCALE = 5
NOOBJECT_SCALE = 1
CLASS_SCALE = 1
COORD_SCALE = 1


"""
hyper-parameter
"""
GPU = ''
LEARNING_RATE = 1e-3                # 学习率
DECAY_STEPS = 30000
DECAY_RATE = 0.1
STAIRCASE = True
BATCH_SIZE = 64                     # batch size
EPOCHS = 30                         # 迭代次数160，论文中为160个迭代，可自定义
SUMMARY_ITER = 10
SAVE_ITER = 1000
MOMENTUM = 0.9                      # 角动量

"""
test parameter
"""
THRESHOLD = 0.2
IOU_THRESHOLD = 0.5                # IOU阈值0.5