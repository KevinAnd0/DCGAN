import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from blocks import DeconvBlock, ConvBlock

def generator():
    model = keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 1024)))    
    model.add(DeconvBlock(512, 5))
    model.add(DeconvBlock(256, 5))
    model.add(DeconvBlock(128, 5))
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def discriminator(img_height, img_width):
    model = keras.Sequential()
    model.add(keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1, input_shape=(img_height, img_width, 3)))
    model.add(ConvBlock(128, 5))
    model.add(ConvBlock(256, 5))
    model.add(ConvBlock(512, 5))
    model.add(ConvBlock(1024, 5))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model