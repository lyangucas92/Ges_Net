import paddle
import math
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from paddle.fluid.initializer import Uniform

train_parameters = {
    "input_size": [3, 224, 224],
    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得
    "label_dict": {},
    "data_dir": "data/data129430/cwtFigure",  # 训练数据存储地址
    "train_file_list": "train.txt",
    "label_file": "label_list.txt",
    "save_freeze_dir": "./freeze-model",
    "save_persistable_dir": "./persistable-params",
    "continue_train": False,        # 是否接着上一次保存的参数接着训练，优先级高于预训练模型
    "pretrained": False,            # 是否使用预训练的模型(不使用预训练模型)
    "pretrained_dir": "data/data131175/SE_ResNext50_32x4d_pretrained", 
    "mode": "train",
    "num_epochs": 20,
    "train_batch_size": 32,
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值
    "use_gpu": True,
    "dropout_seed": None,
    "image_enhance_strategy": {  # 图像增强相关策略
        #"need_distort": True,  # 是否启用图像颜色增强
        "need_rotate": True,   # 是否需要增加随机角度
        "need_crop": True,      # 是否要增加裁剪
        "need_flip": True,      # 是否要增加水平随机翻转
        # "hue_prob": 0.5,
        # "hue_delta": 18,
        # "contrast_prob": 0.5,
        # "contrast_delta": 0.5,
        # "saturation_prob": 0.5,
        # "saturation_delta": 0.5,
        # "brightness_prob": 0.5,
        # "brightness_delta": 0.125
    },
    "early_stop": {
        "sample_frequency": 50,
        "successive_limit": 10,
        "good_acc1": 0.95
    },
    "rsm_strategy": {
        "learning_rate": 0.0001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "momentum_strategy": {
        "learning_rate": 0.001,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "sgd_strategy": {
        "learning_rate": 0.01,
        "lr_epochs": [20, 40, 60, 80, 100],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]
    },
    "adam_strategy": {
        "learning_rate": 0.001
    }
}

class GoogleNet():
    """
    GoogleNet网络类
    """
    def __init__(self):
        self.params = train_parameters

    def conv_layer(self,
                   input,
                   num_filters,
                   filter_size,
                   stride=1,
                   groups=1,
                   act=None,
                   name=None):
        channels = input.shape[1]
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=False,
            name=name)
        return conv

    def xavier(self, channels, filter_size, name):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")

        return param_attr

    def inception(self,
                  input,
                  channels,
                  filter1,
                  filter3R,
                  filter3,
                  filter5R,
                  filter5,
                  proj,
                  name=None):
        conv1 = self.conv_layer(
            input=input,
            num_filters=filter1,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_1x1")
        conv3r = self.conv_layer(
            input=input,
            num_filters=filter3R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3_reduce")
        conv3 = self.conv_layer(
            input=conv3r,
            num_filters=filter3,
            filter_size=3,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3")
        conv5r = self.conv_layer(
            input=input,
            num_filters=filter5R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5_reduce")
        conv5 = self.conv_layer(
            input=conv5r,
            num_filters=filter5,
            filter_size=5,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5")
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=1,
            pool_padding=1,
            pool_type='max')
        convprj = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=proj,
            stride=1,
            padding=0,
            name="inception_" + name + "_3x3_proj",
            param_attr=ParamAttr(
                name="inception_" + name + "_3x3_proj_weights"),
            bias_attr=False)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat

    def net(self, input, class_dim=9):
        conv = self.conv_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act=None,
            name="conv1")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        conv = self.conv_layer(
            input=pool,
            num_filters=64,
            filter_size=1,
            stride=1,
            act=None,
            name="conv2_1x1")
        conv = self.conv_layer(
            input=conv,
            num_filters=192,
            filter_size=3,
            stride=1,
            act=None,
            name="conv2_3x3")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)

        ince3a = self.inception(pool, 192, 64, 96, 128, 16, 32, 32, "ince3a")
        ince3b = self.inception(ince3a, 256, 128, 128, 192, 32, 96, 64,
                                "ince3b")
        pool3 = fluid.layers.pool2d(
            input=ince3b, pool_size=3, pool_type='max', pool_stride=2)

        ince4a = self.inception(pool3, 480, 192, 96, 208, 16, 48, 64, "ince4a")
        ince4b = self.inception(ince4a, 512, 160, 112, 224, 24, 64, 64,
                                "ince4b")
        ince4c = self.inception(ince4b, 512, 128, 128, 256, 24, 64, 64,
                                "ince4c")
        ince4d = self.inception(ince4c, 512, 112, 144, 288, 32, 64, 64,
                                "ince4d")
        ince4e = self.inception(ince4d, 528, 256, 160, 320, 32, 128, 128,
                                "ince4e")
        pool4 = fluid.layers.pool2d(
            input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = self.inception(pool4, 832, 256, 160, 320, 32, 128, 128,
                                "ince5a")
        ince5b = self.inception(ince5a, 832, 384, 192, 384, 48, 128, 128,
                                "ince5b")
        pool5 = fluid.layers.pool2d(
            input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout,
                              size=class_dim,
                              act='softmax',
                              param_attr=self.xavier(1024, 1, "out"),
                              name="out",
                              bias_attr=ParamAttr(name="out_offset"))

        pool_o1 = fluid.layers.pool2d(
            input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = self.conv_layer(
            input=pool_o1,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o1")
        fc_o1 = fluid.layers.fc(input=conv_o1,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o1"),
                                name="fc_o1",
                                bias_attr=ParamAttr(name="fc_o1_offset"))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out1"),
                               name="out1",
                               bias_attr=ParamAttr(name="out1_offset"))

        pool_o2 = fluid.layers.pool2d(
            input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = self.conv_layer(
            input=pool_o2,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o2")
        fc_o2 = fluid.layers.fc(input=conv_o2,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o2"),
                                name="fc_o2",
                                bias_attr=ParamAttr(name="fc_o2_offset"))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out2"),
                               name="out2",
                               bias_attr=ParamAttr(name="out2_offset"))

        # last fc layer is "out"
        return out, out1, out2