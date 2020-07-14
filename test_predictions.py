
import os
import time
import glob
import h5py
import numpy as np
import dask.array as da

import preProcess
import seeSound
import importlib
from networks import asfv

importlib.reload(asfv)
importlib.reload(preProcess)
importlib.reload(seeSound)

dir_ebs = r'/home/ubuntu/avinash/vol_ebs'
dir_hFile = '/home/ubuntu/avinash/vol_ebs/avspeech/storage_20200628'
dir_model = dir_hFile

sound_dir = os.path.join(dir_ebs, 'avspeech/train')
noise_dir = os.path.join(dir_ebs, 'audioset/videos')
dir_temp = os.path.join(dir_ebs, 'temp')


#%% Path to hdf file

path_hFile = glob.glob(dir_hFile + '/stored*.h5')
if len(path_hFile) == 0:
    print('No hdf file found!')
else:
    path_hFile = path_hFile[-1]

#%% Reading from hFile and creating Video Normalizer

with h5py.File(path_hFile, mode='r') as hFile:
    print(hFile.keys())
    print(hFile['vid_samples_train'].shape)
    tic = time.time()
    arr = da.from_array(hFile['vid_samples_train'])
    img_mean = arr.mean(axis=0).mean(axis=-1).compute()
    img_std = arr.std(axis=0).mean(axis=-1).compute()
    vid_norm = preProcess.VideoNormalizer(hFile['vid_samples_train'][:5])
    vid_norm._VideoNormalizer__mean_image = img_mean
    vid_norm._VideoNormalizer__std_image = img_std
    print(int(time.time() - tic), 's')

#%% Save vid_norm
path_vid_norm = os.path.join(dir_hFile, 'vid_norm.pkl')
vid_norm.save(path_vid_norm)


#%% If pre-trained model already exists load from path
tic = time.time()
path_model = glob.glob(dir_model + '/trained_model.h5')
if len(path_model) == 0:
    print('Model not found')
else:
    path_model = path_model[-1]
    neural_network = asfv.NeuralNetwork.load(path_model)
print(int(time.time() - tic), 's')

#%% Load VideoNormalizer
path_vid_norm = os.path.join(dir_hFile, 'vid_norm.pkl')
vid_norm = seeSound.load_vid_norm(path_vid_norm)

#%% Assess predictions
offset = 0
nSlices = 5
inds = range(offset, offset+nSlices+1)
with h5py.File(path_hFile, mode='r') as hFile:
    print(hFile.keys())
    spec_mixed = np.array(hFile['mixed_spectrograms_train'][inds])
    spec_pure = np.array(hFile['sound_spectrograms_train'][inds])
    vid_slices = np.array(hFile['vid_samples_train'][inds])
vid_slices = vid_norm.normalize(vid_slices)

predicted_spectrogram = neural_network.predict(vid_slices, spec_mixed)

predicted_signal = preProcess.reconstruct_speech_signal(out['mixed_signal'],
                                                        predicted_spectrogram, 25)

#%% Save stuff
out['mixed_signal'].save_to_wav_file(os.path.join(dir_temp, 'mixed_signal_norm.wav'))
out['speech_signal'].save_to_wav_file(os.path.join(dir_temp, 'sound_signal_norm.wav'))
predicted_signal.save_to_wav_file(os.path.join(dir_temp, 'predicted_signal_norm.wav'))


#%%
# neural_network = None
# print('Creating model...')
# if neural_network is None:
#     neural_network = \
#         asfv.NeuralNetwork.build(vid_samples_train.shape[1:],
#                                  mixed_spectrograms_train.shape[1:])
# elif isinstance(neural_network, str):
#     if not os.path.exists(neural_network):
#         print('Check path, neural network not found!')
#     else:
#         neural_network = asfv.NeuralNetwork.load(neural_network)
# else:
#     neural_network = asfv.NeuralNetwork(neural_network)


#%% Train from data stored in h5py HDF file

dir_hFile = r'/home/ubuntu/avinash/vol_ebs/avspeech/storage_20200628'
path_hFile = os.path.join(dir_hFile, 'stored_data_20200628.h5')

path_model = seeSound.train_from_hdf(path_hFile, neural_network=None)

#%%