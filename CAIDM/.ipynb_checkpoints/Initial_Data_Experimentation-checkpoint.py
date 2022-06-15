from jarvis.utils.general import gpus

gpus.autoselect()




import numpy as np
from jarvis.train import datasets
from jarvis.utils.display import imshow
from tensorflow.keras import Input
from jarvis.train.client import Client

from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow as tf



def show_mri(mri):
    for i in range(mri.shape[0]):
        clear_output(wait=True)
        plt.axis(False)
        plt.imshow(mri[i, :, :, 0], cmap='gray')
        plt.show()



client = Client('/data/raw/miccai_rsna/data/ymls/client-3d.yml')
gen_train, gen_valid = client.create_generators()



xs, ys = next(gen_train)
# imshow(xs['t2w'][0][24], radius=1)
# print(ys['lbl'][0])


# --- Print keys 
for key, arr in xs.items():
    print('xs key: {} | shape = {}'.format(key.ljust(8), arr.shape))




# --- Create model inputs
inputs = client.get_inputs(Input)




inputs




from tensorflow.keras.layers import Dense, Conv3D, GlobalAveragePooling3D, Flatten, Input, MaxPool3D


base_model = tf.keras.Sequential()
base_model.add(Input(shape=(48, 96, 96, 1)))
base_model.add(Conv3D(16, 3, activation='relu', padding='same'))
base_model.add(MaxPool3D(padding='same'))
base_model.add(Conv3D(32, 3, activation='relu', padding='same'))
base_model.add(MaxPool3D(padding='same'))
base_model.add(Conv3D(48, 3, activation='relu', padding='same'))
base_model.add(MaxPool3D(padding='same'))
base_model.add(Conv3D(64, 3, activation='relu', padding='same'))
base_model.add(MaxPool3D(padding='same'))
base_model.add(Conv3D(80, 3, activation='relu', padding='same'))
base_model.add(MaxPool3D(padding='same'))

base_model.add(GlobalAveragePooling3D())
base_model.add(Dense(128, activation='relu'))
base_model.add(Dense(1, activation='sigmoid'))




base_model.summary()




from tensorflow.keras.losses import BinaryCrossentropy



opt = tf.keras.optimizers.Adam()

@tf.function
def train_step(x, y):
   # print(x.shape)
    with tf.GradientTape() as tape:
        print('beginning training...')
        y_hat = base_model(x, training=True)
        print('done training...')
        bce = BinaryCrossentropy()
        loss = bce(y, y_hat)
    grads = tape.gradient(loss, base_model.trainable_variables)
    opt.apply_gradients(zip(grads, base_model.trainable_variables))
    return loss




def train(train_dataset, n_epochs=10, max_steps_per_epoch=None, adversarial=False):
    for epoch in range(n_epochs):
        for i, batch in enumerate(train_dataset):
            xs, ys = batch
            print(xs['fla'].shape)
            print(ys)
            if max_steps_per_epoch is not None:
                if i == max_steps_per_epoch:
                    break
#             adversarial = (epoch >= 1)
            l1 = train_step(xs['fla'], ys['lbl'].reshape(4, 1))#[:, 0, 0, 0])
#             del batch
            # clear_output(wait=True)
            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            print(f'Step {i+1}')
            print(f'Loss: {round(l1.numpy(), 2)}')
            if adversarial:
                print(f'Discriminator Loss: {round(l2.numpy(), 2)}')




print(ys['lbl'].reshape(4, 1))




print(f'GPU Name: {tf.test.gpu_device_name()}')




# train(gen_train)






