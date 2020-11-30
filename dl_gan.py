
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline
import keras
from tensorflow.python.keras.layers import Dense, Dropout, Input, Reshape, Flatten, Conv2D, BatchNormalization, ReLU, Conv2DTranspose
# from keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.models import Model,Sequential
# from keras.models import Model,Sequential
from tensorflow.python.keras.datasets import mnist, cifar10
# from keras.datasets import mnist, cifar10
from tqdm import tqdm
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
# from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
# from keras.optimizers import Adam
# tf.keras.utils.to_categorical
from keras.utils import plot_model, np_utils

from tensorflow.keras import initializers

from keras import backend as K

# load the dataset
def load_data():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  num_classes = len(np.unique(y_train))
# x_train = (x_train.astype(np.float32) - 127.5)/127.5
# x_train.shape
# convert shape of x_train from (60000, 28, 28) to (60000, 784) 
# 784 columns per row
# x_train = x_train.reshape(50000, 1024,3)
  x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
  x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
  input_shape = (32, 32, 3)

# convert class vectors to binary class matrices
  y_train = np_utils.to_categorical(y_train, num_classes)
  y_test = np_utils.to_categorical(y_test, num_classes)
  return (x_train, y_train, x_test, y_test)
(x_train, y_train,x_test, y_test)=load_data()
print(x_train.shape)

# defune adam optimizer with learning rate of 0.0002
def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

#define the generator and discriminator networks


latent_dim = 100
kernel_init = initializers.RandomNormal(stddev=0.02)

# Generator network
gen = Sequential()

# FC: 2x2x512
gen.add(Dense(2*2*512, input_shape=(latent_dim,), kernel_initializer=kernel_init))
gen.add(Reshape((2, 2, 512)))
gen.add(BatchNormalization())
gen.add(LeakyReLU(0.2))

# # Conv 1: 4x4x256
gen.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
gen.add(BatchNormalization())
gen.add(LeakyReLU(0.2))

# Conv 2: 8x8x128
gen.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
gen.add(BatchNormalization())
gen.add(LeakyReLU(0.2))

# Conv 3: 16x16x64
gen.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
gen.add(BatchNormalization())
gen.add(LeakyReLU(0.2))

# Conv 4: 32x32x3
gen.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                              activation='tanh'))


# gen=Sequential()
# gen.add(Dense(units=256,input_dim=100))
# gen.add(LeakyReLU(0.2))

# gen.add(Dense(units=512))
# gen.add(LeakyReLU(0.2))

# gen.add(Dense(units=1024))
# gen.add(LeakyReLU(0.2))

# gen.add(Dense(units=1024, activation='tanh'))

gen.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
#     return gen
# g=create_gen()
gen.summary()

img_shape = x_train[0].shape

# Discriminator network
desc = Sequential()

# Conv 1: 16x16x64
desc.add(Conv2D(64, kernel_size=5, strides=2, padding='same',
                        input_shape=(img_shape), kernel_initializer=kernel_init))
desc.add(LeakyReLU(0.2))

# Conv 2:
desc.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
desc.add(BatchNormalization())
desc.add(LeakyReLU(0.2))

# Conv 3: 
desc.add(Conv2D(256, kernel_size=5, strides=2, padding='same'))
desc.add(BatchNormalization())
desc.add(LeakyReLU(0.2))

# Conv 3: 
desc.add(Conv2D(512, kernel_size=5, strides=2, padding='same'))
desc.add(BatchNormalization())
desc.add(LeakyReLU(0.2))

# FC
desc.add(Flatten())

# Output
desc.add(Dense(1, activation='sigmoid'))
    # return desc

# d = create_desc()
desc.summary()

desc.compile(optimizer=adam_optimizer(), loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

# d_g = discriminador(generador(z))
desc.trainable = False

z = Input(shape=(latent_dim,))
img = gen(z)
decision = desc(img)
d_g = Model(inputs=z, outputs=decision)

d_g.compile(optimizer=adam_optimizer(), loss='binary_crossentropy',
            metrics=['binary_accuracy'])

d_g.summary()

# trainable parameters
epochs = 100
batch_size = 32
smooth = 0.1

# to compare the results of real and fake samples
real = np.ones(shape=(batch_size, 1))
fake = np.zeros(shape=(batch_size, 1))

d_loss = []
g_loss = []

for e in range(epochs + 1):
    for i in range(len(x_train) // batch_size):
        
        # train discriminator weights
        desc.trainable = True
        
        # Real samples
        x_batch = x_train[i*batch_size:(i+1)*batch_size]
        d_loss_real = desc.train_on_batch(x=x_batch,
                                                   y=real * (1 - smooth))
        
        # Fake Samples
        z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
        x_fake = generator.predict_on_batch(z)
        d_loss_fake = desc.train_on_batch(x=x_fake, y=fake)
         
        # Discriminator loss
        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])
        
        # Train Generator weights
        desc.trainable = False
        g_loss_batch = d_g.train_on_batch(x=z, y=real)

        print(
            'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, i, len(x_train) // batch_size, d_loss_batch, g_loss_batch[0]),
            100*' ',
            end='\r'
        )

    d_loss.append(d_loss_batch)
    g_loss.append(g_loss_batch[0])
    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], g_loss[-1]), 100*' ')

    if e % 10 == 0:
        samples = 10
        x_fake = gen.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))

        for k in range(samples):
            plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
            plt.imshow(((x_fake[k] + 1)* 127).astype(np.uint8))

        plt.tight_layout()
        plt.show()