# -*- coding: utf-8 -*-
# @Time : 2020/10/31 17:12
# @Author : cds
# @Site : https://github.com/SkyLord2?tab=repositories
# @Email: chengdongsheng@outlook.com
# @File : YOLOv2.py
# @Software: PyCharm
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Input, GlobalAveragePooling2D, Softmax, concatenate
from tensorflow.keras.models import Model
import numpy as np

class YOLOv2:
    def __init__(self, alpha = 0.1, anchors = None, classes = None, input_size = 416,  model_path = None, lambda_noobj = None, lambda_prior = None, lambda_coord = None, lambda_obj = None, lambda_class = None):
        self.alpha = alpha
        self.lambda_noobj = lambda_noobj
        self.lambda_prior = lambda_prior
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.anchors = anchors
        self.num_anchor = len(anchors)
        self.classes = classes
        self.num_classes = len(classes)
        self.input_size = input_size
        self.model = self.build(input_size, self.num_classes, self.num_anchor, self.alpha)
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

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad12")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv18", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn18")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad13")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv19", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn19")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad14")(inter_tensor)
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

        inter_tensor = Lambda(lambda x: tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]])), name="pad15")(inter_tensor)
        inter_tensor = Conv2D(filters=1024, kernel_size=3, name="conv21", use_bias=False)(inter_tensor)
        inter_tensor = BatchNormalization(name="bn21")(inter_tensor)
        inter_tensor = LeakyReLU(alpha)(inter_tensor)
        # 输出batchSize * 13 * 13 * (5 + 25)
        # 每个grid cell 分配5个anchor
        # 总共有20个类别
        yolo_out = Conv2D(filters=num_anchor*(num_classe + 5), kernel_size=1, name="conv22", use_bias=False)(inter_tensor)
        # 这里是比较trick的地方，在网络的最后直接通过Lambda层来计算网络的损失，即网络的输出就是网络的损失
        boxes_input = Input(shape=(None, 5))
        detectors_mask_input = Input(shape=(input_size//32, input_size//32, self.num_anchor, 1))
        matching_boxes_input = Input(shape=(input_size//32, input_size//32, self.num_anchor, 5))
        output = Lambda(self.loss, output_shape=(1,), name="yolo_loss", arguments={'anchors': self.anchors, 'num_classes': self.num_classes})([yolo_out, boxes_input, detectors_mask_input, matching_boxes_input])
        # 调整tensor尺寸？？？
        model = Model(inputs=[input_image, boxes_input, detectors_mask_input, matching_boxes_input], outputs=output)
        return model

    def compile_model(self):
        """
        论文中的角动量为0.9，学习率为 1e-3
        :param model: 学习模型
        :return: None
        """
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)
        # 网络的损失已经在网络的最后一层进行计算，网络的输出已经是损失
        self.model.compile(optimizer = optimizer, loss = {"yolo_loss": lambda y_true, y_pred: y_pred}, metrics=['accuracy'])

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
                lr = 1e-4
                if(epoch <= 75):
                    lr = 1e-2
                elif(75 < epoch and epoch <= 105):
                    lr = 1e-3
                elif(105 < epoch and epoch <= 135):
                    lr = 1e-4
                return lr
            learning_scheduler = lr_scheduler
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(learning_scheduler)
        history = self.model.fit(data, labels, batch_size=self.batch_size, epochs=self.epochs, callbacks=[lr_schedule])
        return history
    def train_generator(self, generator, step_per_epoch, epochs, callbacks=None):
        self.model.fit_generator(generator, steps_per_epoch=step_per_epoch, epochs=epochs, callbacks=callbacks)

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

    def preprocess_data_voc(self, gt_boxes, anchors, image_size = 416):
        """
        针对 COCO 数据集做一下预处理

        :param gt_boxes: 标注的真实边界框，形状(batch_size, gt_num, 5)，[[[x_center, y_center, w, h, c]]], 均为使用image_size归一化之后的值，
        要处理为相对于grid_cell尺寸的值
        :param anchors: 对每一个grid cell上使用的anchors（论文中使用了5个），形状（anchor_num, 2）,第二维度表示anchor的宽高
        （与grid cell的壁纸）, [[w , h]], 是grid_cell尺寸的比值
        :param image_size: 输入图像的分辨率 image_size * image_size,
        :return:
        """
        height, width = image_size
        num_anchors = len(anchors)
        # 图片的输入尺寸必须是32的倍数
        assert height % 32 == 0
        assert width % 32 == 0
        # 每一个cell的大小
        height_scale = height//32
        width_scale = width//32

        # 表示一个gt boxes的参数个数， 5
        num_params = gt_boxes.shape[1]
        # 用来负责预测的anchor，0/1 预测/不预测, 形如（13，13，5，1）
        response_anchor = np.zeros((height_scale, width_scale, num_anchors, 1), dtype=np.float32)
        # anchor对于gt box的偏移（相对于grid cell），形如（13，13，5，5）
        matching_true_boxes = np.zeros((height_scale, width_scale, num_anchors, num_params), dtype=np.float32)

        for gt_box in gt_boxes:
            # gt box中对象的类别
            box_class = gt_box[4:5]
            # gt box的坐标[x_center, y_center, w, h]
            box = gt_box[0:4]
            # 坐标使用image_size归一化之后的坐标，要放大到对应的cell
            box = box * np.array([width_scale, height_scale, width_scale, height_scale])
            # x 属于第几个grid cell
            x = np.floor(box[0]).astype('int')
            # y 属于第几个grid cell
            y = np.floor(box[1]).astype('int')

            max_iou = 0
            best_anchor = 0

            # 计算与gt box iou最大的anchors（面积与gt box最接近的anchor，面积大小接近，然后调整位置即可）
            # 将所有anchor和物体的中心点都移到原点再计算iou，这样便于计算
            for k, anchor in enumerate(anchors):
                # gt box的中心在原点
                # 右下角坐标 box_maxs = [w/2, h/2]
                box_maxs = box[2:]/2
                # 左上角
                box_mins = -box_maxs
                # anchor坐标
                anchor_maxs = anchor/2
                anchor_mins = -anchor_maxs
                # 交叉的面积
                inter_mins = np.maximum(box_mins, anchor_mins)
                inter_maxs = np.minimum(box_maxs, anchor_maxs)
                inter_wh = inter_maxs - inter_mins
                inter_area = inter_wh[0]*inter_wh[1]
                # 计算iou
                box_area = box[2]*box[3]
                anchor_area = anchor[0]*anchor[1]
                total_area = box_area + anchor_area - inter_area
                iou = inter_area/total_area

                if(iou > max_iou):
                    max_iou = iou
                    best_anchor = k

            if(max_iou > 0):
                response_anchor[x, y, best_anchor] = 1
                adjusted_box = np.array([
                    box[0] - y,                         # 相对于grid_cell的坐标
                    box[1] - x,
                    np.log(box[2]/anchors[best_anchor][0]),
                    np.log(box[3]/anchors[best_anchor][1]),
                    box_class
                ], dtype=np.float32)
                matching_true_boxes[x, y, best_anchor] = adjusted_box
        return response_anchor, matching_true_boxes

    def preprocess_net_output(self, output, anchors, num_classes):
        """
        转换网络最后一层（不包括损失计算层）的输出[batch_size，height_scale, width_scale, num_anchors, num_classes+5]
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
        anchors_tensor = tf.reshape(anchors, (1, 1, 1, num_anchors, 2))
        conv_dim = output.shape[1:3]

        conv_height_index = tf.reshape(np.arange(start=0, stop=conv_dim[0]),(conv_dim[0],))
        conv_width_index = tf.reshape(np.arange(start=0, stop=conv_dim[1]),(conv_dim[1],))

        conv_height_index = tf.tile(conv_height_index, [conv_dim[1]])
        conv_width_index = tf.tile(tf.expand_dims(conv_width_index, 0), [conv_dim[0], 1])
        conv_width_index = K.flatten(tf.transpose(conv_width_index))
        conv_index = tf.transpose(tf.stack([conv_height_index, conv_width_index]))
        conv_index = tf.reshape(conv_index, (1, conv_dim[0], conv_dim[1], 1, 2))
        conv_index = tf.cast(conv_index, output.dtype)
        # [batch_size, 13, 13, 125] -> [batch_size, 13, 13, 5, 25]
        output = tf.reshape(output, (-1, conv_dim[0], conv_dim[1], num_anchors, num_classes + 5))
        conv_dim = tf.cast(tf.reshape(conv_dim, (1,1,1,1,2)), output.dtype)
        # 缩放到[0, 1]
        box_xy = tf.sigmoid(output[..., :2])
        box_wh = tf.exp(output[..., 2:4])
        box_conf = tf.sigmoid(output[..., 4:5])
        box_cls = tf.nn.softmax(output[..., 5:])

        box_xy = (box_xy + conv_index)/conv_dim
        box_wh = (anchors * box_wh)/conv_dim

        return box_xy, box_wh, box_conf, box_cls
    def loss(self, args, anchors, num_classes):
        """
        计算损失
        :param args:    yolo_out:网络输出(batch_size, 13, 13, 5* (5+num_classes))
                        gt_boxes:真实边界框(batch_size, gt_num, 5)
                        response_anchor: 用来负责预测的anchor，0/1 负责预测/不负责不预测, 形如（batch_size, 13，13，5，1）
                        matching_true_boxes：anchor对于gt box的偏移（相对于grid cell）以及目标类别，形如（batch_size, 13，13，5，5）
        :param anchors: 一组（5个）先验anchor的尺寸
        :param num_classes: 检测目标的类别数
        :return:
        """
        yolo_out, gt_boxes, response_anchor, matching_true_boxes = args
        num_anchors = len(anchors)

        # 各个损失的权重
        obj_scale = 5
        no_obj_scal = 1
        class_scale = 1
        coord_scale = 1
        # pred_xy:[batch_size, 13, 13, 5, 2]
        # pred_wh:[batch_size, 13, 13, 5, 2]
        # pred_conf:[batch_size, 13, 13, 5, 1]
        # pred_cls: [batch_size, 13, 13, 5, 80]
        """
        〇 预处理网络的输出
        """
        pred_xy, pred_wh, pred_conf, pred_cls = self.preprocess_net_output(yolo_out, anchors, num_classes)

        yolo_out_shape = yolo_out.shape
        # shape(batch_size, 13, 13, 5, 25)
        output = tf.reshape(yolo_out, (-1, yolo_out_shape[1], yolo_out_shape[2], num_anchors, num_classes+5))
        '''
        I 计算IOU
        '''
        # 增加维度，以便于和gt_boxes计算交叉面积
        # shape (batch_size, 13, 13, 5, 2) -> (batch_size, 13, 13, 5, 1, 2)
        # batch_szie, height, width, num_anchors, num_gt_boxes, xy/wh
        # 注意：axis=4 和 axis=-1 是不同的
        pred_xy = tf.expand_dims(pred_xy, axis=4)
        pred_wh = tf.expand_dims(pred_wh, axis=4)
        pred_wh_half = pred_wh/2
        # 左上右下角坐标
        # shape (batch_size, 13, 13, 5, 1, 2)
        pred_lu = pred_xy - pred_wh_half
        pred_rd = pred_xy + pred_wh_half
        # 预测框面积
        # shape (batch_size, 13, 13, 5, 1)
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        # 调整gt的形状，便于计算
        # （batch_szie, num_gt_bexes, 5） -> (batch_szie, 1, 1, 5, num_gt_bexes, 5)
        # atch_szie, height, width, num_anchors, num_gt_boxes, (x,y,w,h,c)
        gt_boxes = tf.reshape(gt_boxes, [gt_boxes[0], 1, 1, 1, gt_boxes[1], gt_boxes[2]])
        # shape (batch_szie, 1, 1, 1, num_gt_boxes, 2)
        gt_xy = gt_boxes[..., 0:2]
        gt_wh = gt_boxes[..., 2:4]
        gt_wh_self = gt_wh/2
        # 左上右下角的坐标
        # shape (batch_szie, 1, 1, 1, num_gt_boxes, 2)
        gt_lu = gt_xy - gt_wh_self
        gt_rd = gt_xy - gt_wh_self
        # shape (batch_szie, 1, 1, 1, num_gt_boxes)
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        # 计算交叉区域面积
        # 左上、右下角坐标
        # shape (batch_szie, 13, 13, 5, num_gt_boxes, 2)
        inter_min = tf.maximum(pred_lu, gt_lu)
        inter_max = tf.minimum(pred_rd, gt_rd)
        inter_wh = tf.maximum(inter_max - inter_min, 0.)
        # shape (batch_size, 13, 13, 5, num_gt_boxes)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        union_area = pred_area + gt_area - inter_area
        # shape (batch_size, 13, 13, 5, num_gt_boxes)
        iou = inter_area/union_area
        # shape (batch_size, 13, 13, 5)
        best_iou = tf.reduce_max(iou, axis=4)
        # shape (batch_size, 13, 13, 5, 1)
        # 如果一个anchor中同时与两个待检测目标的iou最大，只检测iou最大的那个目标
        best_iou = tf.expand_dims(best_iou, axis=-1)
        """
        II 计算置信度损失
        负责预测目标的anchor置信度 + 不负责预测目标的anchor置信度
        """
        # shape (batch_size, 13, 13, 5, 1)
        obj_detection = tf.cast(best_iou > 0.6, best_iou.dtype)
        no_obj_detection = 1 - obj_detection
        no_obj_weight = no_obj_scal * no_obj_detection * (1 - response_anchor)
        # 对于不负责预测目标的anchor，其置信度要接近于0
        no_obj_conf_loss = no_obj_weight * tf.square(pred_conf)
        obj_conf_loss = obj_scale * response_anchor * tf.square(pred_conf - best_iou)
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
        matching_boxes = matching_true_boxes[..., 0:4]
        # shape (batch_size, 13, 13, 5, 4)
        coord_loss = coord_scale * response_anchor * tf.square(matching_boxes - pred_boxes)
        """
        IV计算预测类别损失
        """
        # shape (batch_size, 13, 13, 5)
        true_cls = tf.cast(matching_true_boxes[..., 4], tf.float32)
        # shape (batch_size, 13, 13, 5, 20)
        true_cls - tf.one_hot(true_cls, num_classes)
        class_loss = class_scale * response_anchor * tf.square(pred_cls - true_cls)
        """
        V计算总体损失
        """
        conf_loss_sum = tf.reduce_sum(conf_loss)
        coord_loss_sum = tf.reduce_sum(coord_loss)
        class_loss_sum = tf.reduce_sum(class_loss)

        total_loss = 0.5 * (conf_loss_sum + coord_loss_sum + class_loss_sum)

        return total_loss