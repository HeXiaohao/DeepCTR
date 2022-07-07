# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen, weichenswc@163.com

Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)

"""

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Concatenate, Dense

from ..feature_column import build_input_features, get_linear_logit, input_from_feature_columns
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import InteractingLayer
from ..layers.utils import concat_func, add_func, combined_dnn_input


def AutoInt(linear_feature_columns, dnn_feature_columns, att_layer_num=3, att_embedding_size=8, att_head_num=2,
            att_res=True,
            dnn_hidden_units=(256, 128, 64), dnn_activation='relu', l2_reg_linear=1e-5,
            l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_use_bn=False, dnn_dropout=0, seed=1024,
            task='binary', ):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_embedding_size: int.The embedding size in multi-head self-attention network.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    # 边缘参数处理
    if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
        raise ValueError("Either hidden_layer or att_layer_num must > 0")

    # 初始化输入的 dnn 特征的 keras tensor
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    # 这里没有看太懂。感觉是 autoint 输入的线性特征部分，这里直接 * 一个 linear
    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    # 没太看懂这部分用来干什么。处理成系数特征向量 list + 稠密特征 list
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    # 合并当前的稠密向量
    att_input = concat_func(sparse_embedding_list, axis=1)

    # att_layer_num 应该是多头机制里面的头数量，或者说注意力空间的个数
    # 对于每个 fields 的向量，互相计算交叉注意力
    for _ in range(att_layer_num):
        att_input = InteractingLayer(
            att_embedding_size, att_head_num, att_res)(att_input)
    att_output = Flatten()(att_input)

    # 把稀疏特征的 embedding list 和稠密值的 list concat 在一起
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

    # 同时拥有 deep 部分和 Interacting 部分
    if len(dnn_hidden_units) > 0 and att_layer_num > 0:  # Deep & Interacting Layer
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
        stack_out = Concatenate()([att_output, deep_out])
        final_logit = Dense(1, use_bias=False)(stack_out)
    # 只有 deep 部分
    elif len(dnn_hidden_units) > 0:  # Only Deep
        deep_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input, )
        final_logit = Dense(1, use_bias=False)(deep_out)
    # 只有 Interacting 部分
    elif att_layer_num > 0:  # Only Interacting Layer
        final_logit = Dense(1, use_bias=False)(att_output)
    else:  # Error
        raise NotImplementedError

    # 最后把 final_logit 和线性部分结果相加在一起
    final_logit = add_func([final_logit, linear_logit])
    # 预测层
    output = PredictionLayer(task)(final_logit)

    # 构建 model
    model = Model(inputs=inputs_list, outputs=output)

    return model
