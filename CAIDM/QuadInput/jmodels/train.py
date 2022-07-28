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

from tensorflow.keras.layers import Dense, Conv3D, GlobalAveragePooling3D, MaxPool3D, Input, BatchNormalization, Activation, Add, AveragePooling3D, Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras import callbacks
from ResNet10_3D import QuadInput_ResNet, ResNet10_3D
    
    
    
def prep_quad(data):
    xs, ys = data
    inp = (xs['t2w'].reshape(4, 48, 96, 96), xs['t1w'].reshape(4, 48, 96, 96), xs['t1wce'].reshape(4, 48, 96, 96), xs['fla'].reshape(4, 48, 96, 96))
    x_prep = inp
    if(np.isnan(inp[0]).any() or np.isnan(inp[1]).any() or np.isnan(inp[2]).any() or np.isnan(inp[3]).any()):
        x_prep = (np.random.normal(size=(4, 48, 96, 96)), np.random.normal(size=(4, 48, 96, 96)), np.random.normal(size=(4, 48, 96, 96)), np.random.normal(size=(4, 48, 96, 96)))
    return (x_prep, ys['lbl'].reshape(4, 1))


def prep_stacked(data):
    xs, ys = data
    inp = np.empty((4, 48, 96, 96, 4))
    inp = np.stack([xs['t2w'], xs['t1w'], xs['t1wce'], xs['fla']], axis=4)
    inp = inp.reshape((4, 48, 96, 96, 4))
    x_prep = (inp-np.min(inp))/(np.max(inp)-np.min(inp))
    if(np.isnan(x_prep).any()):
        x_prep = np.random.normal(size=(4, 48, 96, 96, 4))
    return (x_prep, ys['lbl'].reshape(4, 1))
        
def lr_decay(epochs, lr):
    lr *= p['LR_decay']
    return lr


gpus.autoselect()


p = params.load("./hyper.csv")


base_model = QuadInput_ResNet()
bce = tf.keras.losses.BinaryCrossentropy()
base_model.compile(optimizer=tf.keras.optimizers.Adam(p['LR']), loss=bce, metrics=['acc'])





log_dir = p['output_dir']
tb_callback = callbacks.TensorBoard(log_dir)

lr_callback = callbacks.LearningRateScheduler(lr_decay)


client = Client('/data/raw/miccai_rsna/data/ymls/client-3d.yml')
gen_train, gen_valid = client.create_generators()
MODEL_NAME = '{}/model.h5'.format(p['output_dir'])

history = model.fit(x=map(prep_quad, gen_train),
                         steps_per_epoch=100,
                         epochs=p['epochs'],
                         validation_data=map(prep_quad, gen_valid),
                         validation_steps=100,
                         validation_freq=4,
                         callbacks=[tb_callback, lr_callback])

# --- Save model
model.save(MODEL_NAME)