import preProcess
import readWriteMedia
import time
import importlib
import os
import glob
# from nmfTools.NMFtoolbox import forwardSTFT
importlib.reload(preProcess)
importlib.reload(readWriteMedia)

movDir = '/home/avi/Downloads/temp'
movPaths = glob.glob(os.path.join(movDir, '*.mp4'))
# movPath = os.path.join(movDir, movName)
# print(movPaths)

inds = [1, 2]
speech_file_path = movPaths[inds[0]]
noise_file_path = movPaths[inds[1]]

out = preProcess.preprocess_video_pair(speech_file_path, noise_file_path)
out['noise_signal'].save_to_wav_file(os.path.join(movDir, 'noise.wav'))
out['speech_signal'].save_to_wav_file(os.path.join(movDir, 'speech.wav'))
out['mixed_signal'].save_to_wav_file(os.path.join(movDir, 'mixed.wav'))


"""
# yt_url = r'https://www.youtube.com/watch?v=gMz4r5RLvKs'
# yt_url = r'https://www.youtube.com/watch?v=fregObNcHC8'
# yt_url = yt_url.split('?v=')[-1]
"""

