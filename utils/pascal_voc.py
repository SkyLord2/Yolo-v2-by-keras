# -*- coding: utf-8 -*-
# @Time : 2020/10/31 17:12
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : pascal_voc.py
# @Software: PyCharm
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg

"""
└── VOCdevkit     #根目录
    └── VOC2007   #不同年份的数据集，这里只下载了2012的，还有2007等其它年份的
        ├── Annotations        #存放xml文件，与JPEGImages中的图片一一对应，解释图片的内容等等
        ├── ImageSets          #该目录下存放的都是txt文件，txt文件中每一行包含一个图片的名称，末尾会加上±1表示正负样本
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation
        ├── JPEGImages         #存放源图片
        ├── SegmentationClass  #存放的是图片，语义分割相关
        └── SegmentationObject #存放的是图片，实例分割相关
"""


class pascal_voc(object):
    def __init__(self, phase, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')  # data/pascal_voc/VOCdevkit(数据集的根目录，包含不同年份的数据)
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')  # data/pascal_voc/VOCdevkit/VOC2007
        print("file path=", os.path.abspath(__file__))
        self.max_boxes_num = 42                                     # voc 2007中一张图片最多有42个bounding box
        self.cache_path = cfg.CACHE_PATH  # data/pascal_voc/cache
        self.batch_size = cfg.BATCH_SIZE  # batch size 为 64
        self.image_size = cfg.IMAGE_SIZE  # image_size为 448*448
        self.image_scale = 32
        self.classes = cfg.CLASSES  # pascal_voc的20个类别
        self.anchors = cfg.YOLO_ANCHORS
        self.num_anchors = len(self.anchors)
        self.num_classes = len(self.classes)
        self.cell_size = cfg.CELL_SIZE  # 整张图片的cell_size*cell_size, 即7*7
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))  # 将类转化为索引字典
        self.flipped = cfg.FLIPPED  # 是否反转true
        self.phase = phase  # 训练或者测试，字符串"train"
        self.rebuild = rebuild  # False
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.prepare()  # 获得gt boxes的label包含了经过水平翻转后的训练实例

    def get(self):
        """
        获取到batch_size(64)大小的图片和对应的gt boxes label
        :return images, labels:
        """
        return self.get_by_size(self.batch_size)

    def get_all(self):
        """
        获取所有的图片和对应的label
        :return:
        """
        total_size = len(self.gt_labels)
        return self.get_by_size(total_size)

    def get_by_size(self, size):
        """
        获取指定大小的图片和对应的label
        :param size:
        :return:
        """
        images = np.zeros(
            (size, self.image_size, self.image_size, 3), dtype=np.float32)
        matching_true_boxes = np.zeros(
            (size, self.cell_size, self.cell_size, self.num_anchors, 5), dtype=np.float32)
        response_anchors = np.zeros(
            (size, self.cell_size, self.cell_size, self.num_anchors, 1), dtype=np.float32)
        gt_boxes = np.zeros(
            (size, self.max_boxes_num, 5), dtype=np.float32)
        count = 0
        while count < size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)  # 读取图片
            matching_true_boxes[count, :, :, :, :] = self.gt_labels[self.cursor]['label']  # 图片的label
            response_anchors[count, :, :, :, :] = self.gt_labels[self.cursor]['responseanchor']
            gt_boxes[count, :, :] = self.gt_labels[self.cursor]['gtbox']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)  # 随机打乱gt_labels
                self.cursor = 0
                self.epoch += 1  # epoch加1
        return images, matching_true_boxes, gt_boxes, response_anchors  # 返回size大小的图片和对应的label，注意其shape

    def image_read(self, imname, flipped=False):
        """
        读取图片
        :param imname:      路径
        :param flipped:     是否反转
        :return:
        """
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))  # resize大小为416
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0)
        if flipped:
            image = image[:, ::-1, :]  # 反转图片
        return image

    def prepare(self):
        gt_labels = self.load_labels()  # 加载gt_boxes
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')  # 附加水平翻转训练实例
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):  # 循环一次处理一张图片
                gt_labels_cp[idx]['flipped'] = True  # 原来的false变成true，即翻转
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]  # 第二个维度倒序排列，步长为1

                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        for k in range(self.num_anchors):
                            if (gt_labels_cp[idx]['label'][i, j, k] != 0).any():  # 图片反转过之后，中心点坐标改变
                                gt_labels_cp[idx]['label'][i, j, k, 1] = 1 - gt_labels_cp[idx]['label'][i, j, k, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels  # 包含有经过水平翻转后的图片，包含经过水平翻转的训练实例

    def load_labels(self):
        """
        加载标签文件，获取标签
        :return: gt_labels: List
        """
        cache_file = os.path.join(
            self.cache_path,
            'pascal_' + self.phase + '_gt_labels.pkl')  # data/pascal_voc/cache/pascal_/train/_gt_labels.pkl

        if os.path.isfile(cache_file) and not self.rebuild:  # 从cache文件中读取labels
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)  # data/pascal_voc/VOCdevkit/VOC2007

        if not os.path.exists(self.cache_path):  # 如果不存在这个文件则创建
            os.makedirs(self.cache_path)

        if self.phase == 'train':  # train
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')  # 加载trainval.txt文件
        else:  # test
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]  # 加载得到图片的索引，以供train或者test

        gt_labels = []
        print("number of gt labels is %d" % len(self.image_index))

        self.max_boxes_num = 0

        for i, index in enumerate(self.image_index):
            print("parsing gt label, index:%d" % (i))  # 得到train或者test文件中所有指向图片的index,
            label, gt_box, reponse_anchor, box_num = self.load_pascal_annotation(index)  # 读取annotation文件, return label and len(objs)
            if box_num == 0:
                continue

            if(box_num > self.max_boxes_num):
                self.max_boxes_num = box_num

            imname = os.path.join(self.data_path, 'JPEGImages',
                                  index + '.jpg')  # data/pascal_voc/VOCdevkit/VOC2007/JPEGImages/index.jpg
            gt_labels.append({'imname': imname,  # 图片的路径
                              'label': label,  # shape(cell_size, cell_size, 4+1)
                              'gtbox': gt_box,
                              'responseanchor': reponse_anchor,
                              'flipped': False})

        print("max boxes num in image:%d" % (self.max_boxes_num))

        for idx, gt_label in enumerate(gt_labels):
            box = gt_label["gtbox"]
            box_ = np.zeros([self.max_boxes_num, 5], dtype=np.float32)
            box_[:box.shape[0]] = box
            gt_label["gtbox"] = box_

        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:  # 保存 label 到 cache file
            pickle.dump(gt_labels, f)
        return gt_labels  # 返回label，以及该index文件中object的数量

    def load_pascal_annotation(self, index):
        """
        从一个 XML 文件中加载 一张图片中所有 边界框 坐标
        :param index: 图片索引
        :return:
        """
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')  # data/pascal_voc/VOCdevkit/VOC2007/Annotations/index.xml
        tree = ET.parse(filename)  # 解析xml文件

        size_img = tree.find('size')
        im_width = float(size_img.find('width').text)
        im_height = float(size_img.find('height').text)

        h_ratio = 1.0 * self.image_size / im_height  # 416所占图片高度的比例
        w_ratio = 1.0 * self.image_size / im_width  # 416占图片宽度的比例
        # im = cv2.resize(im, [self.image_size, self.image_size])
        # shape (cell_size, cell_size, 1-confidence + 4-coordinate + 20-classs_one_hot)
        label = np.zeros((self.cell_size, self.cell_size, self.num_anchors, 5), dtype=np.float32)  # label数组维度 (image_size//32)*(image_size//32)*25
        gt_box = []
        reponse_anchors = np.zeros((self.cell_size, self.cell_size, self.num_anchors, 1), dtype=np.float32)

        objs = tree.findall('object')  # 找到index指向的该xml文件中的所有object

        for obj in objs:  # 解析xml文件中边界框的坐标
            bbox = obj.find('bndbox')

            x1 = max((float(bbox.find('xmin').text) - 1), 0)  # 像素索引从零开始
            y1 = max((float(bbox.find('ymin').text) - 1), 0)  # 范围[0, 13]
            x2 = max((float(bbox.find('xmax').text) - 1), 0)
            y2 = max((float(bbox.find('ymax').text) - 1), 0)

            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]  # 将类别转化为索引
            # 原始图片尺寸下的坐标
            y_center = (y2 + y1) / 2.0
            x_center = (x2 + x1) / 2.0
            w = x2 - x1
            h = y2 - y1

            # 将坐标转换到新的尺寸，416*416
            x_center_ = x_center * w_ratio / self.image_scale
            y_center_ = y_center * h_ratio / self.image_scale
            w_ = w * w_ratio / self.image_scale
            h_ = h * h_ratio / self.image_scale

            y_ind = np.floor(y_center_).astype('int')  # y_center落在哪个cell, cell的数量为13*13
            x_ind = np.floor(x_center_).astype('int')  # x_center落在哪个cell
            box_l = [x_center_, y_center_, w_, h_]                              # 排列的方式为(x, y, w, h),loss函数中计算交并比时要注意
            box = np.array(box_l, dtype=np.float32)
            best_iou = 0.
            best_anchor = 0

            for k, anchor in enumerate(self.anchors):  # 计算anchors中与gt box的大小最接近的anchor
                box_maxs = box[2:] / 2.0  # 将anchor和gt box的中心平移到远点计算交叉面积
                box_mins = -box_maxs
                anchor_maxs = anchor / 2.0
                anchor_mins = -anchor_maxs
                inter_mins = np.maximum(box_mins, anchor_mins)
                inter_maxs = np.minimum(box_maxs, anchor_maxs)
                inter_wh = np.maximum(inter_maxs - inter_mins, 0)
                inter_area = inter_wh[0] * inter_wh[1]
                box_area = box[2] * box[3]
                anchor_area = anchor[0] * anchor[1]
                total_area = anchor_area + box_area - inter_area
                iou = inter_area / total_area

                if (iou > best_iou):
                    best_iou = iou
                    best_anchor = k
            if (best_iou > 0):
                reponse_anchors[y_ind, x_ind, best_anchor] = 1
                coord_param = [
                    box[0] - x_ind, # x
                    box[1] - y_ind, # y
                    np.log(box[2] / self.anchors[best_anchor][0]),  # w
                    np.log(box[3] / self.anchors[best_anchor][1]),  # h
                    cls_ind]
                adjust_box = np.array(coord_param, dtype=np.float32)
                gt_box.append([
                    x_center / im_width,
                    y_center / im_height,
                    w / im_width,
                    h / im_height,
                    cls_ind
                ])
                label[y_ind, x_ind, best_anchor] = adjust_box
        gt_box = np.array(gt_box, dtype=np.float32)
        return label, gt_box, reponse_anchors, len(objs)