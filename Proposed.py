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
class RAT():
    def __init__(self, layers=50):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=9):
        layers = self.layers  #layers = 50
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name='conv1', ) #7×7×64 卷积核conv1
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max') #3×3最大池化
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name="conv1", )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max')
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv3')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max')
        block0 = 0
        for i in range(depth[block0]):
            conv = self.bottleneck_block(
                input=conv,
                num_filters=num_filters[block0],
                stride=2 if i == 0 and block0 != 0 else 1, 
                cardinality=cardinality,
                reduction_ratio=reduction_ratio
                )
        block1 = 1
        for i in range(depth[block1]):
            conv = self.bottleneck_block(
                input=conv,
                num_filters=num_filters[block1],
                stride=2 if i == 0 and block1 != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio,
                )
        pool1 = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=4,
                pool_padding=1,
                pool_type='max') #3×3最大池化
        out1 = pool1 * 0.1
        block2 = 2
        for i in range(depth[block2]):
            conv = self.bottleneck_block(
                input=conv,
                num_filters=num_filters[block2],
                stride=2 if i == 0 and block2 != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio
                )
        pool2 = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max') #3×3最大池化
        out2 = pool2 * 0.1
        block3 = 3
        for i in range(depth[block3]):
            conv = self.bottleneck_block1(
                input=conv,
                num_filters=num_filters[block3],
                stride=2 if i == 0 and block3 != 0 else 1,
                cardinality=cardinality
                )
        out3 = conv * 0.8
        cat = fluid.layers.concat(input=[out1,out2,out3], axis=1) 
        # SE
        se = self.squeeze_excitation(
            input=cat,
            num_channels=cat.shape[1],
            reduction_ratio=16,
            ) # (-1, 3072, 7, 7)
        pool = fluid.layers.pool2d(
            input=se, pool_size=7, pool_type='avg', global_pooling=True)
        drop = fluid.layers.dropout(
            x=pool, dropout_prob=0.5, seed=self.params['dropout_seed'])
        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            act="softmax",
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))
        return out

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(
                input, ch_out, filter_size, stride)
        else:
            return input

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride,
                         cardinality,
                         reduction_ratio,
                         name=None):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            #name='conv' + name + '_x1'
            ) #1×1卷积核
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
            #name='conv' + name + '_x2'
            ) #3×3卷积核
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            #name='conv' + name + '_x3'
            ) #1×1卷积核
        # 通道注意力和空间注意力并行
        scale1 = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
            #name='fc' + name
            )
        sam = SAM_Module()
        scale2 = sam(conv2)  
        scale3 = fluid.layers.elementwise_add(x=scale1, y=scale2) 

        short = self.shortcut(input, num_filters * 2, stride, name=name)

        return fluid.layers.elementwise_add(x=short, y=scale3, act='relu') #输入 x 与输入 y 逐元素相加，并将各个位置的输出元素保存到返回结果中

    def bottleneck_block1(self,
                        input,
                        num_filters,
                        stride,
                        cardinality,
                        name=None):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu'
            ) #1×1卷积核 
        pasm = PSAModule(conv0.shape[1])
        conv1 = pasm(conv0)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            ) #1×1卷积核
        short = self.shortcut(input, num_filters * 2, stride, name=name)
        
        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu') #输入 x 与输入 y 逐元素相加，并将各个位置的输出元素保存到返回结果中
    
    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr())
        return fluid.layers.batch_norm(
             input=conv,
             act=act,
             param_attr=ParamAttr(),
             bias_attr=ParamAttr()
             )
    #通道注意力SE
    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                ),
            bias_attr=ParamAttr())
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                ),
            bias_attr=ParamAttr())
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0) #输入 x 与输入 y 逐元素相乘，并将各个位置的输出元素保存到返回结果中，SE的关键部分
        return scale

# 空间注意力机制CA
class SAM_Module(nn.Layer):  
    def __init__(self):  
        super().__init__()  
        self.conv_after_concat = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)  
        self.sigmoid_spatial = nn.Sigmoid()  

    def forward(self, x):  
        # Spatial Attention Module  
        module_input = x  
        avg = paddle.mean(x, axis=1, keepdim=True)   #沿 axis 计算 x 的平均值。
        mx = paddle.argmax(x, axis=1, keepdim=True)  #沿 axis 计算输入 x 的最大元素的索引
        mx = paddle.cast(mx, 'float32')
        x = paddle.concat([avg, mx], axis=1)
        x = self.conv_after_concat(x)  
        x = self.sigmoid_spatial(x)  
        x = fluid.layers.elementwise_mul(x=module_input,y=x,axis=0)

        return x 


class PSAModule(nn.Layer):
    def __init__(self, in_channels, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super().__init__()
        self.conv_1 = nn.Conv2D(in_channels,in_channels//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0],bias_attr=False)#3×3卷积stride=1
        self.conv_2 = nn.Conv2D(in_channels, in_channels//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1],bias_attr=False)#5×5卷积stride=1
        self.conv_3 = nn.Conv2D(in_channels, in_channels//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2],bias_attr=False)#7×7卷积stride=1
        self.conv_4 = nn.Conv2D(in_channels, in_channels//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3],bias_attr=False)#9×9卷积stride=1
        # self.se = SEWeightModule(in_channels // 4)
        self.se = SEWeightModule(in_channels)
        self.sam = SAM_Module()
        # self.sam = SAM_module(in_channels // 4)
        # self.conv_5 = nn.Conv2D(in_channels // 2, in_channels, kernel_size=1,stride=1,bias_attr=False)#1×1卷积stride=1
        
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        # x1_se = self.se(x1)
        # x2_se = self.se(x2)
        # x3_se = self.se(x3)
        # x4_se = self.se(x4)
        # x_se = paddle.concat([x1_se, x2_se, x3_se, x4_se], axis=1)
        # x_sam = self.sam(x_se)

        x1_sam1 = self.sam(x1)
        x2_sam2 = self.sam(x2)
        x3_sam3 = self.sam(x3)
        x4_sam4 = self.sam(x4)
        x_sam = paddle.concat([x1_sam1, x2_sam2, x3_sam3, x4_sam4], axis=1)
        x_se = self.se(x_sam)
        return x_se

class SEWeightModule(nn.Layer):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight