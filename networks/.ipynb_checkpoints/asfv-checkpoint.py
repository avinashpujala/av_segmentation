"""
Audio-Segmentaion-From-Video (asfv) network. Neural network for isolating sound
sources from regions of interest in videos with audio tracks
"""
import numpy as np

from keras.layers import Input, Dense, Conv2D, Convolution2D 
from keras.layers import MaxPooling2D, Deconvolution2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 
from keras.callbacks import TensorBoard
from keras import optimizers


def audio_encoder(audio_input, batch_norm_kwargs={}, leaky_relu_kwargs={}):
    """
    Convolutional Neural Network (CNN) that returns an embedding from audio input.
    Parameters
    ----------
    audio_input: array
    batch_norm_kwargs: dict
        Dictionary of batch normalization parameters to be applied before leaky ReLu
        for each layer of the CNN. If given {}, uses defaults. 
        See keras.layers.BatchNormalization.
    leaky_relu_kwargs: dict
        Leaky ReLu activation to be applied to units/neurons. See 
        keras.layers.LeakyRelu
    Returns
    -------
    nn: audio encoder network    
    """
    bn = BatchNormalization(**batch_norm_kwargs)
    lr = LeakyReLU(**leaky_relu_kwargs)
    
    x = Convolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
    x = lr(bn(x))
    
    x = Convolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
    x = lr(bn(x))
    
    x = Convolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = lr(bn(x))
    
    x = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = lr(bn(x))
    
    x = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = lr(bn(x))
    
    return x


def audio_decoder(embedding, batch_norm_kwargs={}, leaky_relu_kwargs={}):
    """
    Decodes audio input from embedding returned by encoder; in that sense
    it's the inverse function for audio_encoder
    Parameters
    ----------
    See audio_encoder
    Returns
    -------
    nn: audio_decoder network
    """    
    bn = BatchNormalization(**batch_norm_kwargs)
    lr = LeakyReLU(**leaky_relu_kwargs)
    x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(embedding)
    x = lr(bn(x))
    
    x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = lr(bn(x))
    
    x = Deconvolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = lr(bn(x))
    
    x = Deconvolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
    x = lr(bn(x))
    
    x = Deconvolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = lr(bn(x))
    x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    
    return x


def video_encoder(video_input, dropout=0.3, batch_norm_kwargs={}, leaky_relu_kwargs={}):
    """
    Encoder for embedding video input
    Parameters
    ----------
    batch_norm_kwargs, leaky_relu_kwargs: see audio_encoder
    dropout: scalar
        Unit dropout rate for each layer. Typically, higher
        dropout rates are used for input layers, but here it's
        uniform across layers
    Returns
    -------
    nn: video encoder network
    """
    bn = BatchNormalization(**batch_norm_kwargs)
    lr = LeakyReLU(**leaky_relu_kwargs)
    
    x = Convolution2D(128, kernel_size=(5, 5), padding='same')(video_input)
    x = lr(bn(x))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(128, kernel_size=(5, 5), padding='same')(x)
    x = lr(bn(x))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
    x = lr(bn(x))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
    x = lr(bn(x))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
    x = lr(bn(x))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
    x = lr(bn(x))
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    return x