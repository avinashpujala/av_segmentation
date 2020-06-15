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


def audio_encoder(audio_input, batch_norm_kwargs=dict(), leaky_relu_kwargs=dict()):
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


def audio_decoder(embedding, batch_norm_kwargs=dict(), leaky_relu_kwargs=dict()):
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


def video_encoder(video_input, dropout=0.3, batch_norm_kwargs=dict(), leaky_relu_kwargs=dict()):
    """
    Encoder for embedding video input
    Parameters
    ----------
    video_input: array
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


class NeuralNetwork(object):
    def __init__(self, model):
        self.__model = model
        self.__batch_norm_kwargs=batch_norm_kwargs
        self.__leaky_relu_kwwargs=leaky_relu_kwargs

    @classmethod
    def __build_encoder(cls, vid_shape, aud_spec_shape_ext, batch_norm_kwargs=dict(),
                        leaky_relu_kwargs=dict()):
        """

        Parameters
        ----------
        vid_shape: tuple
            Shape of video input
        aud_spec_shape_ext: tuple
            Shape of extended audio spectrogram input.
            If audio spectrogram shape = (nTimePts, nFreqScales, timeWinSize), then extended audio
            spectrogram shape = (nTimePts, nFreqScales, timeWinSize, 1), i.e. channel dimension added
        batch_norm_kwargs: dict
            Batch normalization parameters. See keras.layers.BatchNormalization
        leaky_relu_kwargs: dict
            Leaky relu parameters. See keras.layers.LeakyRelu
        Returns
        -------
        model: keras neural netwok model
            Combined video & audio encoder model that takes video and audio inputs
            inputs = [video_input, audio_input], and returns video_audio_embedding
        combined_embedding_size: tuple
            Combined video-audio embedding size = (video_embedding size + audio_embedding_size)//4
        audio_embedder_shape: list
            Shape of the audio embedder excluding the batch size
        """
        vid_input = Input(shape=vid_shape)
        aud_input = Input(shape=aud_spec_shape_ext)

        vid_embedder = video_encoder(vid_input)
        aud_embedder = audio_encoder(aud_input)

        vid_embedding = Flatten()(vid_embedder)
        aud_embedding = Flatten()(aud_embedder)

        x = concatenate([vid_embedding, aud_embedding])
        combined_embedding_size = int(x._keras_shape[1]/4)

        x = Dense(combined_embedding_size)(x)
        x = BatchNormalization(**batch_norm_kwargs)(x)
        vid_aud_embedding = LeakyReLU(**leaky_relu_kwargs)(x)

        model = Model(inputs=[vid_input, aud_input], outputs=vid_aud_embedding)
        model.summary()
        return model, combined_embedding_size, aud_embedder.shape[1:].as_list()


    @classmethod
    def __build_decoder(cls, combined_embedding_size, aud_embedding_shape,
                        batch_norm_kwargs=dict(), leaky_relu_kwargs=dict()):
        bn = BatchNormalization(**batch_norm_kwargs)
        lr = LeakyReLU(**leaky_relu_kwargs)

        combined_embedding_input = Input(shape=(combined_embedding_size, ))
        x = Dense(combined_embedding_size)(combined_embedding_input)
        x = lr(bn(x))

        aud_emedding_size = np.prod(aud_embedding_shape)
        x = Dense(aud_emedding_size)(x)
        x = Reshape(aud_embedding_shape)(x)
        aud_embedding = lr(bn(x))
        aud_output = audio_decoder(aud_embedding, batch_norm_kwargs=batch_norm_kwargs,
                                   leaky_relu_kwargs=leaky_relu_kwargs)
        model = Model(inputs=combined_embedding_input, outputs=aud_output)
        model.summary()
        return model

    @classmethod
    def build(cls, vid_shape, aud_spec_shape, optimizer='adam', learning_rate=5e-4):
        """

        Parameters
        ----------
        vid_shape
        aud_spec_shape
        optimizer: str
            Name of optimizer to use, e.g. 'adam' or 'rmsprop'. At present, did not test beyond
            these two optimizers, so proceed with caution, if using other optimizer.
        learning_rate: scalar
            Learning rate for optimizer

        Returns
        -------

        """
        aud_spec_shape_ext = list(aud_spec_shape).append(1) # Add channel dim

        encoder, combined_embedding_size, aud_embedding_shape = \
            cls.__build_encoder(vid_shape, aud_spec_shape_ext)

        decoder = cls.__build_decoder(combined_embedding_size, aud_embedding_shape)

        vid_input = Input(shape=vid_shape)
        aud_input = Input(shape=aud_spec_shape_ext)

        aud_output = decoder(encoder([vid_input, aud_input]))

        opt = eval(f'optimizers.{optimizer}(lr={learning_rate})')
        model = Model(inputs=[vid_input, aud_input], outputs=aud_output)
        model.compile(loss='mean_squared_error', optimizer=opt)

        model.summary()
        return NeuralNetwork(model)


