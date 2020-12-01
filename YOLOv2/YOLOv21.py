# -*- coding: utf-8 -*-
# @Time : 2020/10/31 17:12
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : YOLOv2.py
# @Software: PyCharm
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import config as cfg
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Input, GlobalAveragePooling2D, Softmax, concatenate
from tensorflow.keras.models import Model
import numpy as np

class YOLOv2:
    def __init__(self, alpha = 0.1, model_path = None):
        self.alpha = alpha
        self.model_path = model_path
        self.batch_size = cfg.BATCH_SIZE
        self.epochs = cfg.EPOCHS
        self.learning_rate = cfg.LEARNING_RATE
        self.momentum = cfg.MOMENTUM
        self.lambda_noobj = cfg.NOOBJECT_SCALE
        self.lambda_coord = cfg.COORD_SCALE
        self.lambda_obj = cfg.OBJECT_SCALE
        self.lambda_class = cfg.CLASS_SCALE
        self.anchors = cfg.YOLO_ANCHORS
        self.num_anchor = len(self.anchors)
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.input_size = cfg.IMAGE_SIZE
        self.conv_index = self.generateOffsetGrid([self.input_size // 32, self.input_size // 32], tf.float32)
        self.model = self.build(self.input_size, self.num_classes, self.num_anchor, self.alpha)
        # 加载权重
        if(model_path is not None):
            self.model_load(model_path)

    def build(self, input_size, num_classe, num_anchor = 5, alpha = 0.1):
        """
        构建DarkNet-19
        :param num_classes: 检测目标类别数量
        :param alpha: leaky relu 的激活系数
        :return model:  网络模型
        """
        def PassThrough(x):
            return tf.compat.v1.space_to_depth(x, block_size=2)
        keras.backend.clear_session()
        # 输入尺寸随迭代变化
        input_image = Input(shape=(input_size, input_size, 3), dtype="float32")
        # pad size: kernel_size//2
        # 使用BN层之后，卷积层没有偏置
        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0],[1, 1],[1, 1],[0, 0]])), name="pad1")(input_image)
        inter_tensor = Conv2D(filters=32, kernel_size=3, name="conv1", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn1")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = MaxPool2D(pool_size=(2, 2), strides=2)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad2")(inter_tensor)
        inter_tensor = Conv2D(filters=64, kernel_size=3, name="conv2", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn2")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = MaxPool2D(pool_size=(2, 2), strides=2)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad3")(inter_tensor)
        inter_tensor = Conv2D(filters=128, kernel_size=3, name="conv3", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn3")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Conv2D(filters=64, kernel_size=1, name="conv4", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn4")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad5")(inter_tensor)
        inter_tensor = Conv2D(filters=128, kernel_size=3, name="conv5", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn5")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = MaxPool2D(pool_size=(2, 2), strides=2)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad6")(inter_tensor)
        inter_tensor = Conv2D(filters=256, kernel_size=3, name="conv6", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn6")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Conv2D(filters=128, kernel_size=1, name="conv7", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn7")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad7")(inter_tensor)
        inter_tensor = Conv2D(filters=256, kernel_size=3, name="conv8", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn8")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = MaxPool2D(pool_size=(2, 2), strides=2)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad8")(inter_tensor)
        inter_tensor = Conv2D(filters=512, kernel_size=3, name="conv9", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn9")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Conv2D(filters=256, kernel_size=1, name="conv10", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn10")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad9")(inter_tensor)
        inter_tensor = Conv2D(filters=512, kernel_size=3, name="conv11", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn11")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Conv2D(filters=256, kernel_size=1, name="conv12", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn12")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad10")(inter_tensor)
        inter_tensor = Conv2D(filters=512, kernel_size=3, name="conv13", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn13")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        # passthrough layer 分支
        pass_through = inter_tensor

        inter_tensor = MaxPool2D(pool_size=(2, 2), strides=2)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad11")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv14", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn14")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Conv2D(filters=512, kernel_size=1, name="conv15", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn15")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad12")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv16", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn16")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Conv2D(filters=512, kernel_size=1, name="conv17", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn17")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad13")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv18", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn18")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad14")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv19", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn19")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad15")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv20", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn20")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        # passthrough layer
        pass_through = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pass_through_pad")(pass_through)
        pass_through = Conv2D(filters=64, kernel_size=3, name="pass_through_conv", use_bias=False)(pass_through)
        pass_through = BatchNormalization(name="pass_through_bn")(pass_through)
        pass_through = LeakyReLU(alpha)(pass_through)
        pass_through = Lambda(PassThrough)(pass_through)

        inter_tensor = concatenate([pass_through, inter_tensor])

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad16")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv21", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn21")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)
        # 输出batchSize * 13 * 13 * (5 + 25)
        # 每个grid cell 分配5个anchor
        # 总共有20个类别
        yolo_out = Conv2D(filters=num_anchor*(num_classe + 5), kernel_size=1, name="conv22", use_bias=False)(inter_tensor)
        # 这里是比较trick的地方，在网络的最后直接通过Lambda层来计算网络的损失，即网络的输出就是网络的损失
        gt_boxes_input = Input(shape=(input_size//32, input_size//32, self.num_anchor, 5))
        response_anchor_input = Input(shape=(input_size//32, input_size//32, self.num_anchor, 1))
        output = Lambda(self.loss, output_shape=(1,), name="yolo_loss", arguments={'anchors': self.anchors, 'num_classes': self.num_classes})([yolo_out, gt_boxes_input, response_anchor_input])
        # 构建模型
        model = Model(inputs=[input_image, gt_boxes_input, response_anchor_input], outputs=output)
        return model

    def compile_model(self):
        """
        论文中的角动量为0.9，学习率为 1e-3
        :param model: 学习模型
        :return: None
        """
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum, clipnorm=1.)
        # 网络的损失已经在网络的最后一层进行计算，网络的输出已经是损失
        self.model.compile(optimizer = optimizer, loss = {"yolo_loss": lambda y_true, y_pred: y_pred}, metrics=['mse'])

    # YOLO_ANCHORS = np.array(((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),(7.88282, 3.52778), (9.77052, 9.16828)))
    # anchors_value = np.array([[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]], dtype='float32')

    def train(self, data, labels, learning_scheduler = None):
        """
        :param data: 训练数据
        :param labels: 标签
        :return:
        """
        if(learning_scheduler is None):
            def lr_scheduler(epoch):
                lr = 1e-3
                if (epoch <= 60):
                    lr = 1e-3
                elif (60 < epoch and epoch <= 90):
                    lr = 1e-4
                elif (90 < epoch and epoch <= 135):
                    lr = 1e-5
                return lr
            learning_scheduler = lr_scheduler
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(learning_scheduler)
        history = self.model.fit(data, labels, batch_size=self.batch_size, epochs=self.epochs, callbacks=[lr_schedule])
        return history
    def train_generator(self, generator, data_size, callbacks=None):
        if(callbacks is None):
            def lr_scheduler(epoch):
                lr = 1e-3
                if (epoch <= 60):
                    lr = 1e-3
                elif (60 < epoch and epoch <= 90):
                    lr = 1e-4
                elif (90 < epoch and epoch <= 135):
                    lr = 1e-5
                return lr
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
            callbacks = [lr_schedule]
        self.model.fit_generator(generator(), steps_per_epoch=data_size//self.batch_size, epochs=self.epochs, callbacks=callbacks)

    def model_summary(self):
        self.model.summary()

    def model_save(self, path):
        """
        保存模型
        :param path: 保存路径 yolo_v1_model.h5
        :return:
        """
        self.model.save_weights(path)

    def model_load(self, path):
        self.model.load_weights(path)

    def generateOffsetGrid(self, conv_dim, dtype):
        """
        生成网格的偏执
        :param conv_dim: x,y方向的网格数量
        :param dtype: 数据类型
        :return: conv_index
        """
        conv_height_index = tf.reshape(np.arange(start=0, stop=conv_dim[0]), (conv_dim[0],),
                                       name="generateOffsetGrid_reshape_conv_height_index")
        conv_width_index = tf.reshape(np.arange(start=0, stop=conv_dim[1]), (conv_dim[1],),
                                      name="generateOffsetGrid_reshape_conv_width_index")

        conv_height_index = tf.tile(conv_height_index, [conv_dim[1]])
        conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0), [conv_dim[0], 1])
        conv_width_index = K.flatten(tf.transpose(conv_width_index))
        conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
        conv_index = tf.reshape(conv_index, (1, conv_dim[0], conv_dim[1], 1, 2), name="generateOffsetGrid_reshape_conv_index")
        conv_index = tf.cast(conv_index, dtype, name="generateOffsetGrid_cast_conv_index")
        return conv_index

    def preprocess_net_output(self, output, anchors, num_classes):
        """
        转换网络最后一层（不包括损失计算层）的输出[batch_size，height_scale, width_scale, num_anchors, num_classes+5]，还原到正常的坐标宽高
        :param output: 网络最后一层的输出，不包括损失计算层
        :param anchors: 每个grid cell采用5个anchor，[[w,h]]，w，h是与grid cell尺寸的比值
        :param num_classes: 类别数量
        :return:
                box_xy
                box_wh
                box_conf
                box_cls
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.cast(anchors, output.dtype, name="preprocess_cast_anchord"), (1, 1, 1, num_anchors, 2), name="preprocess_reshape_anchors")
        conv_dim = output.shape[1:3]

        conv_index = self.conv_index
        # [batch_size, 13, 13, 125] -> [batch_size, 13, 13, 5, 25]
        output = tf.reshape(output, (-1, conv_dim[0], conv_dim[1], num_anchors, num_classes + 5), name="preprocess_reshape_output")
        conv_dim = tf.cast(tf.reshape(conv_dim, (1,1,1,1,2), name="preprocess_reshape_conv_dim"), output.dtype, name="preprocess_cast_conv_dim")
        # 使用sigmoid缩放到[0, 1]，是因维中心坐标只能在grid cell之内
        # (y,x,h,w)
        box_xy = tf.sigmoid(output[..., :2])
        box_wh = tf.exp(output[..., 2:4])
        box_conf = tf.sigmoid(output[..., 4:5])
        box_cls = tf.nn.softmax(output[..., 5:])
        # 加上grid cell的偏置，除以conv_dim，进行归一化
        box_xy = (box_xy + conv_index)/conv_dim
        # 论文的公式
        box_wh = (anchors_tensor * box_wh)/conv_dim

        return box_xy, box_wh, box_conf, box_cls

    def preprocess_gt_boxes(self, gt_boxes, anchors):
        """
        对gt_boxes进行预处理，还原到正常的坐标宽高
        :param gt_boxes:
        :param anchors:
        :return:
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.cast(anchors, gt_boxes.dtype, name="preprocess_gt_boxes_cast_anchord"),
                                    (1, 1, 1, num_anchors, 2), name="preprocess_gt_boxes_reshape_anchors")
        conv_dim = gt_boxes.shape[1:3]

        conv_index = self.conv_index

        conv_dim = tf.cast(tf.reshape(conv_dim, (1, 1, 1, 1, 2), name="preprocess_gt_boxes_reshape_conv_dim"), gt_boxes.dtype,
                           name="preprocess_gt_boxes_cast_conv_dim")
        # (y,x,h,w)
        box_xy = gt_boxes[..., :2]
        box_wh = tf.exp(gt_boxes[..., 2:4])
        box_cls = gt_boxes[..., 4:]
        # 加上grid cell偏置，除以conv_dim，进行归一化
        box_xy = (box_xy + conv_index) / conv_dim
        box_wh = (anchors_tensor * box_wh) / conv_dim

        return box_xy, box_wh, box_cls
    def loss(self, args, anchors, num_classes):
        """
        计算损失
        :param args:    yolo_out:网络输出(batch_size, 13, 13, 5* (5+num_classes)), 归一化之后的值
                        gt_boxes:真实边界框(batch_size, 13, 13, 5, 5) (y,x,h,w)， 归一化之后的值
                        response_anchor: 用来负责预测的anchor，0/1 负责预测/不负责不预测, 形如（batch_size, 13，13，5，1）
                        matching_true_boxes：anchor对于gt box的偏移（相对于grid cell）以及目标类别，形如（batch_size, 13，13，5，5）
        :param anchors: 一组（5个）先验anchor的尺寸
        :param num_classes: 检测目标的类别数
        :return:
        """
        yolo_out, gt_boxes, response_anchor = args
        num_anchors = len(anchors)

        # 各个损失的权重
        obj_scale = self.lambda_obj
        no_obj_scal = self.lambda_noobj
        class_scale = self.lambda_class
        coord_scale = self.lambda_coord
        # pred_xy:[batch_size, 13, 13, 5, 2]
        # pred_wh:[batch_size, 13, 13, 5, 2]
        # pred_conf:[batch_size, 13, 13, 5, 1]
        # pred_cls: [batch_size, 13, 13, 5, 20]
        """
        〇 预处理网络的输出
        """
        pred_xy, pred_wh, pred_conf, pred_cls = self.preprocess_net_output(yolo_out, anchors, num_classes)

        yolo_out_shape = yolo_out.shape
        # shape(batch_size, 13, 13, 5, 25)
        output = tf.reshape(yolo_out, (-1, yolo_out_shape[1], yolo_out_shape[2], num_anchors, num_classes+5), name="loss_reshape_ouput")
        '''
        I 计算IOU
        '''
        # 增加维度，以便于和gt_boxes计算交叉面积
        # shape (batch_size, 13, 13, 5, 2)
        # batch_szie, height, width, num_anchors, xy/wh
        pred_wh_half = pred_wh/2
        # 左上右下角坐标
        # shape (batch_size, 13, 13, 5, 2)
        pred_lu = pred_xy - pred_wh_half
        pred_rd = pred_xy + pred_wh_half
        # 宽度与高度
        # shape(batch_size, 13, 13, 5)
        pred_h = pred_wh[..., 0]
        pred_w = pred_wh[..., 1]
        # 恢复形状
        # shape(batch_size, 13, 13, 5, 1)
        pred_w = tf.expand_dims(pred_w, axis=-1)
        pred_h = tf.expand_dims(pred_h, axis=-1)
        # 预测框面积
        # shape (batch_size, 13, 13, 5, 1)
        pred_area = pred_w * pred_h

        # shape (batch_szie, 13, 13, 5, 2)
        # batch_szie, height, width, num_anchors, (x,y,w,h,c)
        gt_xy, gt_wh, gt_cls = self.preprocess_gt_boxes(gt_boxes, anchors)
        gt_wh_self = gt_wh/2
        # 左上右下角的坐标
        # shape (batch_szie, 13, 13, 5, 2)
        gt_lu = gt_xy - gt_wh_self
        gt_rd = gt_xy + gt_wh_self
        # 宽度与高度
        # shape (batch_szie, 13, 13, 5)
        gt_h = gt_wh[..., 0]
        gt_w = gt_wh[..., 1]
        # 恢复维度
        # shape (batch_szie, 13, 13, 5, 1)
        gt_w = tf.expand_dims(gt_w, axis=-1)
        gt_h = tf.expand_dims(gt_h, axis=-1)
        # shape (batch_szie, 13, 13, 5, 1)
        gt_area = gt_w * gt_h
        # 计算交叉区域面积
        # 左上、右下角坐标
        # shape (batch_szie, 13, 13, 5, 2)
        inter_min = tf.maximum(pred_lu, gt_lu)
        inter_max = tf.minimum(pred_rd, gt_rd)
        inter_wh = tf.maximum(inter_max - inter_min, 0.)
        # 交叉区域的宽度与高度
        # shape (batch_size, 13, 13, 5)
        inter_h = inter_wh[..., 0]
        inter_w = inter_wh[..., 1]
        # 恢复维度
        # shape(batch_size, 13, 13, 5, 1)
        inter_w = tf.expand_dims(inter_w, axis=-1)
        inter_h = tf.expand_dims(inter_h, axis=-1)
        inter_area = inter_w * inter_h
        union_area = pred_area + gt_area - inter_area
        # shape (batch_size, 13, 13, 5, 1)
        iou = inter_area/union_area
        """
        II 计算置信度损失
        负责预测目标的anchor置信度 + 不负责预测目标的anchor置信度
        """
        # shape (batch_size, 13, 13, 5, 1)
        obj_detection = tf.cast(iou > 0.6, iou.dtype, name="loss_cast_iou")
        no_obj_detection = 1 - obj_detection

        # shape (batch_size, 13, 13, 5, 1)
        no_obj_weight = no_obj_scal * no_obj_detection * (1 - response_anchor)
        # 对于不负责预测目标的anchor，其置信度要接近于0
        no_obj_conf_loss = no_obj_weight * tf.square(pred_conf)
        obj_conf_loss = obj_scale * response_anchor * tf.square(pred_conf - iou)
        # shape (batch_size, 13, 13, 5, 1)
        conf_loss = no_obj_conf_loss + obj_conf_loss
        """
        III计算坐标损失
        负责预测的anchor的坐标损失
        不负责预测的anchors的坐标损失？？？？
        """
        # 得到x,y,w,h坐标用于计算坐标损失
        # shape (batch_size, 13, 13, 5, 4)
        pred_boxes = tf.concat([tf.sigmoid(output[..., 0:2]), output[..., 2:4]], axis=-1)
        matching_boxes = gt_boxes[..., 0:4]
        # shape (batch_size, 13, 13, 5, 4)
        coord_loss = coord_scale * response_anchor * tf.square(matching_boxes - pred_boxes)
        """
        IV计算预测类别损失
        """
        # shape (batch_size, 13, 13, 5)
        true_cls = gt_boxes[..., 4]
        # shape (batch_size, 13, 13, 5, 20)
        true_class = tf.one_hot(tf.cast(true_cls, tf.int32), num_classes)
        # shape (batch_size, 13, 13, 5, 20)
        class_loss = class_scale * response_anchor * tf.square(pred_cls - true_class)
        """
        V计算总体损失
        """
        conf_loss_sum = tf.reduce_sum(conf_loss, axis=[1,2,3,4])
        coord_loss_sum = tf.reduce_sum(coord_loss, axis=[1,2,3,4])
        class_loss_sum = tf.reduce_sum(class_loss, axis=[1,2,3,4])

        total_loss = 0.5 * (conf_loss_sum + coord_loss_sum + class_loss_sum)
        total_loss = tf.reduce_sum(total_loss, keepdims=True)
        tf.print("total_loss", total_loss, output_stream=sys.stderr)
        return total_loss