import preProcess
import seeSound
import importlib
import os
import time

from networks import asfv

importlib.reload(asfv)
importlib.reload(preProcess)
importlib.reload(seeSound)

dir_ebs = r'/home/ubuntu/avinash/vol_ebs'
sound_dir = os.path.join(dir_ebs, 'avspeech/train')
noise_dir = os.path.join(dir_ebs, 'audioset/videos')
dir_temp = os.path.join(dir_ebs, 'temp')

#%%
# sound_paths, noise_paths = seeSound.get_file_paths(dir_speech, noise_dir=None)
tic = time.time()
sound_paths, noise_paths = seeSound.get_file_paths(sound_dir, noise_dir=noise_dir,
                                                   n_files=10)
sound_paths.append('Blah.mp4')
noise_paths = list(noise_paths)
noise_paths.append('Blah.mp4')

print('Processing..')
out = preProcess.preprocess_video_pair(sound_paths, noise_paths, verbose=True)
# path_model = seeSound.train(sound_dir, noise_dir=noise_dir, n_files=5000)
print(int(time.time()-tic), 's')