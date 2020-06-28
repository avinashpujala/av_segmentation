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

    x = Convolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Convolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Convolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)

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
    x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(embedding)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Deconvolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Deconvolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    
    x = Deconvolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    
    return x


def video_encoder(video_input, dropout=0.3, batch_norm_kwargs=dict(),
                  leaky_relu_kwargs=dict()):
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

    x = Convolution2D(128, kernel_size=(5, 5), padding='same')(video_input)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(128, kernel_size=(5, 5), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    
    x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(**batch_norm_kwargs)(x)
    x = LeakyReLU(**leaky_relu_kwargs)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(dropout)(x)
    return x


class NeuralNetwork(object):
    def __init__(self, model):
        self.__model = model

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
        x = BatchNormalization(**batch_norm_kwargs)(x)
        x = LeakyReLU(**leaky_relu_kwargs)(x)

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
        aud_spec_shape_ext = list(aud_spec_shape)
        aud_spec_shape_ext.append(1)  # Add ch dimension

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

    def train(self, train_mixed_spectrograms, train_video_samples,
              train_speech_spectrograms, validation_mixed_spectrograms,
              validation_video_samples, validation_speech_spectrograms,
              model_cache_path, tensorboard_dir, **train_kwargs):
        train_mixed_spectrograms = np.expand_dims(train_mixed_spectrograms, -1)  # expand channels axis
        train_speech_spectrograms = np.expand_dims(train_speech_spectrograms, -1)

        validation_mixed_spectrograms = np.expand_dims(validation_mixed_spectrograms, -1)
        validation_speech_spectrograms = np.expand_dims(validation_speech_spectrograms, -1)

        verbose = train_kwargs.get('verbose', 1)
        checkpoint = ModelCheckpoint(model_cache_path, verbose=verbose,
                                     save_best_only=True)
        lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                     min_lr=0, verbose=verbose)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01,
                                       patience=10, verbose=verbose)
        tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                                  write_graph=True, write_images=True)
        train_kwargs['batch_size'] = train_kwargs.get('batch_size', 16)
        train_kwargs['epochs'] = train_kwargs.get('epochs', 1000)
        train_kwargs['verbose'] = train_kwargs.get('verbose', 1)
        self.__model.fit(x=[train_video_samples, train_mixed_spectrograms],
                         y=train_speech_spectrograms,
                         validation_data=([validation_video_samples,
                                           validation_mixed_spectrograms],
                                          validation_speech_spectrograms),
                         callbacks=[checkpoint, lr_decay, early_stopping, tensorboard],
                         **train_kwargs)

    def predict(self, video_samples, mixed_spectrograms):
        mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
        speech_spectrograms = self.__model.predict([video_samples, mixed_spectrograms])
        return np.squeeze(speech_spectrograms)

    def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):
        mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)
        speech_spectrograms = np.expand_dims(speech_spectrograms, -1)
        loss = self.__model.evaluate(x=[video_samples, mixed_spectrograms],
                                     y=speech_spectrograms)

        return loss

    @staticmethod
    def load(model_cache_path):
        model = load_model(model_cache_path)
        return NeuralNetwork(model)

    def save(self, model_cache_path):
        self.__model.save(model_cache_path)


