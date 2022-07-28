import numpy as np
from jarvis.train import datasets
from jarvis.utils.display import imshow
from tensorflow.keras import Input
from jarvis.train.client import Client
from jarvis.utils.general import gpus
from jarvis.train import params

from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv3D, GlobalAveragePooling3D, MaxPool3D, Input, BatchNormalization, Activation, Add, AveragePooling3D

from tensorflow.keras import callbacks

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
    
        
def lr_decay(epochs, lr):
    lr *= p['LR_decay']
    return lr


gpus.autoselect()


p = params.load("./hyper.csv")


base_model = ResNet10_3D()
bce = tf.keras.losses.BinaryCrossentropy()
base_model.compile(optimizer=tf.keras.optimizers.Adam(p['LR']), loss=bce, metrics=['acc'])


def prep(data):
    xs, ys = data
    inp = np.empty((4, 48, 96, 96, 4))
    inp = np.stack([xs['t2w'], xs['t1w'], xs['t1wce'], xs['fla']], axis=4)
    inp = inp.reshape((4, 48, 96, 96, 4))
    x_prep = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
    if(np.isnan(x_prep).any()):
        x_prep = np.random.normal(size=(4, 48, 96, 96, 4))
    return (x_prep, ys['lbl'].reshape(4, 1))


log_dir = p['output_dir']
tb_callback = callbacks.TensorBoard(log_dir)

lr_callback = callbacks.LearningRateScheduler(lr_decay)


# --- Prepare dataset
paths = datasets.download(name='miccai-rsna')

# --- Alternative method for looking up paths
paths = jtools.get_paths('miccai-rsna')

CLIENT_TEMPLATE = '{}/data/ymls/client-3d.yml'.format(paths['code'])
CLIENT_TRAINING = '{}/client.yml'.format(p['output_dir'])
MODEL_NAME = '{}/model.hdf5'.format(p['output_dir'])

# --- prepare generators
# client = Client('/data/raw/miccai_rsna/data/ymls/client-3d.yml')
# gen_train, gen_valid = client.create_generators()

# --- Prepare client
client = prepare_client(paths, p)
client.load_data_in_memory()
gen_train, gen_valid = client.create_generators()

history = base_model.fit(x=map(prep, gen_train),
                         steps_per_epoch=100,
                         epochs=p['epochs'],
                         callbacks=[tb_callback, lr_callback])

# --- Save model
model.save(MODEL_NAME)

# --- Save client
client.to_yml(CLIENT_TRAINING)