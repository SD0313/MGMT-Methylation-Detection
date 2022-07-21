import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv3D, GlobalAveragePooling3D, MaxPool3D, Input, BatchNormalization, Activation, Add, AveragePooling3D
import numpy as np


def build_resnet():

    inp = Input(shape=(48, 96, 96, 4))
    c1 = Conv3D(kernel_size=7, filters=64, strides=2, padding='same')(inp)
    bn1 = BatchNormalization()(c1)
    a1 = Activation('relu')(bn1)
    p1 = MaxPool3D()(a1)

    #residual block 1
    x = Conv3D(kernel_size=3, filters=64, strides=1, padding='same')(p1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(kernel_size=3, filters=64, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    r2 = Add()([p1, x])

    #residual block 2
    x = Conv3D(kernel_size=3, filters=128, strides=1, padding='same')(r2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(kernel_size=3, filters=128, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y = Conv3D(kernel_size=1, filters=128, strides=1, padding='same')(r2)
    y = Activation('relu')(y)

    r3 = Add()([y, x])

    #residual block 3
    x = Conv3D(kernel_size=3, filters=256, strides=1, padding='same')(r3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(kernel_size=3, filters=256, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y = Conv3D(kernel_size=1, filters=256, strides=1, padding='same')(r3)
    y = Activation('relu')(y)

    r4 = Add()([y, x])

    #residual block 4
    x = Conv3D(kernel_size=3, filters=512, strides=1, padding='same')(r4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(kernel_size=3, filters=512, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y = Conv3D(kernel_size=1, filters=512, strides=1, padding='same')(r4)
    y = Activation('relu')(y)

    r5 = Add()([y, x])

    flat = GlobalAveragePooling3D()(r5)
    out = Dense(1, activation='sigmoid')(flat)


    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


class ResNet10_3D(tf.keras.Model):
    def __init__(self):
        super(ResNet10_3D, self).__init__()
        self.model = build_resnet()
    
    def call(self, xs):
        return self.model(xs)
    
#     def compile(self, **kwargs):
#         super(ResNet10_3D, self).compile()
#         self.opt = kwargs['optimizer']
#         self.loss = kwargs['loss']
    
#     def train_step(self, data):
#         xs, ys = data
#         with tf.GradientTape() as tape:
#             y_hat = self(xs, training=True)
#             loss = self.loss(ys, y_hat)
#         grads = tape.gradient(loss, self.trainable_variables)
#         self.opt.apply_gradients(zip(grads, self.trainable_variables))
#         return {'loss': loss}
    
#     def test_step(self, data):
#         xs, ys = data
#         y_pred = self(xs, training=False)
#         loss = self.loss(ys, y_pred)
#         return {'loss': loss}
    
    
            