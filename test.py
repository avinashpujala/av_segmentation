import preProcess
import seeSound
import importlib
import os
import time
import glob

from networks import asfv

importlib.reload(asfv)
importlib.reload(preProcess)
importlib.reload(seeSound)

dir_ebs = r'/home/ubuntu/avinash/vol_ebs'
sound_dir = os.path.join(dir_ebs, 'avspeech/train')
noise_dir = os.path.join(dir_ebs, 'audioset/videos')
dir_temp = os.path.join(dir_ebs, 'temp')

#%% If pre-trained model already exists load from path
tic = time.time()
path_model = '/home/ubuntu/avinash/vol_ebs/avspeech/storage_20200623/trained_model.h5'
neural_network = asfv.NeuralNetwork.load(path_model)
print(int(time.time() - tic), 's')

#%%
# sound_paths, noise_paths = seeSound.get_file_paths(dir_speech, noise_dir=None)
tic = time.time()
print('Processing..')
path_model = seeSound.train(sound_dir, noise_dir=noise_dir, n_files=500,
                            neural_network=neural_network)
print(int(time.time()-tic), 's')

#%% Load model
# net = asfv.load_model(path_model)
my_net = asfv.NeuralNetwork.load(path_model)
path_vid_norm = os.path.split(path_model)[0]
path_vid_norm = glob.glob(path_vid_norm + '/*.pkl')[-1]
vid_norm = seeSound.load_vid_norm(path_vid_norm)

#%% Assess predictions
ind = 10
# sound_paths, noise_paths = seeSound.get_file_paths(sound_dir, noise_dir=noise_dir,
#                                                    n_files=100)
out = preProcess.preprocess_video_pair(sound_paths[ind], noise_paths[ind],
                                       relevant_only=False)
vid_norm.normalize(out['vid_slices'])
predicted_spectrogram = my_net.predict(out['vid_slices'], out['mixed_spectrograms'])

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


#%%