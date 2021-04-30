from tensorflow import keras
from tensorflow.keras import layers

def DeconvBlock(filters, ks):
    model = keras.Sequential()
    model.add(layers.Conv2DTranspose(filters=filters, kernel_size=ks, strides=2, padding='same', activation='relu', use_bias=False))
    model.add(layers.BatchNormalization())
    return model

def ConvBlock(filters, ks):
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=filters, kernel_size=ks, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))
    return model