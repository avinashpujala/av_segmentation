import preProcess
import readWriteMedia
import time
import importlib
import os
importlib.reload(preProcess)

movDir = '/home/avi/Downloads'
movName = '--80gIqjPgs_215.982433_220.286733.mp4'
movPath = os.path.join(movDir, movName)

# mov_stack, fps = preProcess.preprocess_video_sample(movPath)
aud, vid = readWriteMedia.separate_streams(movPath)

aud_signal = readWriteMedia.AudioSignal(aud.ts, aud.fps)
