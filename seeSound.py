import numpy as np
import glob
from os.path import join, split
from os.path import exists as path_exists
from os import mkdir
import pickle
import h5py

import preProcess
from networks import asfv
from util import util


def get_file_paths(sound_dir, noise_dir=None, n_files=None, ext='mp4'):
    if noise_dir is None:
        noise_dir = sound_dir
    ext = ext.split('.')[-1]
    sound_paths = glob.glob(join(sound_dir, f'*.{ext}'))
    if n_files is None:
        n_files = len(sound_paths)
    print(f'{n_files} file paths')
    noise_paths = glob.glob(join(noise_dir, f'*.{ext}'))
    noise_paths = np.union1d(noise_paths, sound_paths)
    np.random.shuffle(noise_paths)
    np.random.shuffle(sound_paths)
    noise_paths = noise_paths[:len(sound_paths)]
    return sound_paths[:n_files], noise_paths[:n_files]


def train(sound_dir, noise_dir=None, neural_network=None, n_files=None, val_split=0.15, ext='mp4',
          **train_kwargs):
    """
    Train model using a simpler pipeline that is mostly automated for the user.
    The user simply has to point to the directory with pure sound videos and directory
    with noise-containing videos
    Parameters
    ----------
    sound_dir: str
        Directory containing the video to train on. The videos must contain a single sound
        and the object generating the sound (e.g. face, musical instrument, etc)
    noise_dir: str or None
        Directory with videos containing audio noise. If None, then uses audio from other
        videos in sound_dir
    n_files: int
        Number of video files to read from. If None, reads all files in sound_dir. This
        number exists to restrict training to a smaller random subset because of memory
        constraints
    neural_network: None, str, or asfv.NeuralNework class
        If None, then creates a new model. If str then path to saved model, else
        asssumes that the loaded model itself has been given as input.
    val_split: scalar
        Fraction of samples to use for validation
    ext: str
        Extension of video files in sound_dir and noise_dir
    train_kwargs: dict
        Keyword arguments for model.fit, where model is keras neural network model
    Returns
    -------
    path_model: str
        Path to trained model
    """
    print('Getting paths...')
    sound_paths, noise_paths = get_file_paths(sound_dir,
                                              noise_dir=noise_dir,
                                              n_files=n_files, ext=ext)
    dir_store = util.apply_recursively(lambda p: split(p)[0], sound_paths[0])
    dir_store = join(dir_store, f'storage_{util.timestamp("day")}')
    if not path_exists(dir_store):
        mkdir(dir_store)
    dir_tensorboard = join(dir_store, 'tensorboard_logdir')
    path_model = join(dir_store, 'trained_model.h5')

    print('Extracting data from paths...')
    out = preProcess.preprocess_video_pair(sound_paths, noise_paths, verbose=True)
    inds_del = [i for i in range(len(out)) if out[i] is None]
    if len(inds_del) > 0:
        print(f'Could not read {inds_del} files!')
    mixed_spectrograms, vid_samples, sound_spectrograms = \
        map(lambda x: np.delete(x, inds_del, axis=0), out)
    mixed_spectrograms, vid_samples, sound_spectrograms = \
        map(np.array, (mixed_spectrograms, vid_samples, sound_spectrograms))
    n_samples = mixed_spectrograms.shape[0]
    inds_all = np.arange(n_samples)
    np.random.shuffle(inds_all)
    n_samples_val = int(n_samples*val_split)
    inds_val = inds_all[:n_samples_val]
    inds_train = inds_all[n_samples_val:]
    print(f'{len(inds_val)}/{len(inds_all)} validation samples')
    mixed_spectrograms_train, vid_samples_train, sound_spectrograms_train = \
        mixed_spectrograms[inds_train], vid_samples[inds_train], \
        sound_spectrograms[inds_train]
    mixed_spectrograms_val, vid_samples_val, sound_spectrograms_val = \
        mixed_spectrograms[inds_val], vid_samples[inds_val], \
        sound_spectrograms[inds_val]

    print('Normalizing videos..')
    vid_normalizer = preProcess.VideoNormalizer(vid_samples_train)
    vid_normalizer.normalize(vid_samples_train)
    vid_normalizer.normalize(vid_samples_val)

    print('Saving normalizer...')
    path_normalizer = join(dir_store, 'normalizer.pkl')
    with open(path_normalizer, mode='wb') as norm_file:
        pickle.dump(vid_normalizer, norm_file)

    print('Creating model...')
    if neural_network is None:
        neural_network = \
            asfv.NeuralNetwork.build(vid_samples_train.shape[1:],
                                     mixed_spectrograms_train.shape[1:])
    elif isinstance(neural_network, str):
        if not path_exists(neural_network):
            print('Check path, neural network not found!')
        else:
            neural_network = asfv.NeuralNetwork.load(neural_network)

    print('Training model...')
    train_kwargs['verbose'] = train_kwargs.get('verbose', 1)
    train_kwargs['epochs'] = train_kwargs.get('epochs', 1000)
    train_kwargs['batch_size'] = train_kwargs.get('batch_size', 16)
    neural_network.train(mixed_spectrograms_train, vid_samples_train,
                         sound_spectrograms_train, mixed_spectrograms_val,
                         vid_samples_val, sound_spectrograms_val, path_model,
                         dir_tensorboard, **train_kwargs)

    print('Saving model...')
    neural_network.save(path_model)
    return path_model


def train_from_hdf(path_hFile, neural_network=None, n_samples_per_block=1000,
                   n_epochs_per_block=20, n_epoch_cycles=100, val_split=0.2,
                   **train_kwargs):
    """
    Train model from data stored in HDF file (h5py)
    Parameters
    ----------
    path_hFile: str
        Path to hdf file
    neural_network: None, str, or asfv.NeuralNework class
        If None, then creates a new model. If str then path to saved model, else
        asssumes that the loaded model itself has been given as input.
    n_samples_per_block: int
        Number of training samples per block of training
    n_epochs_per_block: int
        Number of epochs per_block
    n_epoch_cycles: int
        Number of epoch cycles
    val_split: scalar
        Fraction of samples to use for validation
    train_kwargs: dict
        Keyword arguments for model.fit, where model is keras neural network model
    Returns
    -------
    path_model: str
        Path to trained model
    """

    dir_store = split(path_hFile)[0]

    if not path_exists(dir_store):
        mkdir(dir_store)
    dir_tensorboard = join(dir_store, 'tensorboard_logdir')
    path_model = join(dir_store, 'trained_model.h5')

    with h5py.File(path_hFile, mode='r') as hFile:
        print('Creating model...')
        if neural_network is None:
            neural_network = \
                asfv.NeuralNetwork.build(hFile['vid_samples_train'].shape[1:],
                                         hFile['mixed_spectrograms_train'].shape[1:])
        elif isinstance(neural_network, str):
            if not path_exists(neural_network):
                print('Check path, neural network not found!')
            else:
                neural_network = asfv.NeuralNetwork.load(neural_network)
        train_kwargs['verbose'] = train_kwargs.get('verbose', 1)
        train_kwargs['epochs'] = train_kwargs.get('epochs', n_epochs_per_block)
        train_kwargs['batch_size'] = train_kwargs.get('batch_size', 16)

        n_samples_train_total = hFile['mixed_spectrograms_train'].shape[0]
        inds_train_all = np.arange(n_samples_train_total)
        n_samples_val_total = hFile['mixed_spectrograms_val'].shape[0]
        inds_val_all = np.arange(n_samples_val_total)
        for iCycle in range(n_epoch_cycles):
            print(f'Epoch cycle {iCycle+1}/{n_epoch_cycles}')
            initial_epoch = iCycle * n_epochs_per_block
            np.random.shuffle(inds_train_all)
            np.random.shuffle(inds_val_all)
            inds_train = inds_train_all[:n_samples_per_block]
            n_val = int(n_samples_per_block * val_split)
            inds_val = inds_val_all[:n_val]
            inds_train, inds_val = np.sort(inds_train), np.sort(inds_val)
            print('Reading data for block...')
            print('Mixed spectrograms')
            mixed_spectrograms_train = hFile['mixed_spectrograms_train'][inds_train]
            print('Video samples...')
            vid_samples_train = hFile['vid_samples_train'][inds_train]
            print('Sound spectograms..')
            sound_spectrograms_train = hFile['sound_spectrograms_train'][inds_train]

            mixed_spectrograms_val = hFile['mixed_spectrograms_val'][inds_val]
            vid_samples_val = hFile['vid_samples_val'][inds_val]
            sound_spectrograms_val = hFile['sound_spectrograms_val'][inds_val]

            print('Normalizing videos..')
            vid_normalizer = preProcess.VideoNormalizer(vid_samples_train)
            vid_normalizer.normalize(vid_samples_train)
            vid_normalizer.normalize(vid_samples_val)

            print('Training...')
            train_kwargs['initial_epoch'] = train_kwargs.get('initial_epoch',
                                                             initial_epoch)
            neural_network.train(mixed_spectrograms_train, vid_samples_train,
                                 sound_spectrograms_train, mixed_spectrograms_val,
                                 vid_samples_val, sound_spectrograms_val, path_model,
                                 dir_tensorboard, **train_kwargs)

    print('Saving model...')
    neural_network.save(path_model)
    return path_model





def load_vid_norm(path):
    with open(path, mode='rb') as vid_file:
        vid_norm = pickle.load(vid_file)
    return vid_norm

