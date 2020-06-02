
import os
import inOut
import importlib
importlib.reload(inOut)

dataDir = '/media/avi/LaCie/Insight/project/AVSpeech datasets'
file_csv = 'avspeech_train.csv'
path_csv = os.path.join(dataDir, file_csv)
path_data = inOut.download_av_speech(path_csv, out_dir='train1')
# vidinfos = inOut.download_av_speech(path_csv, out_dir='/home/avi/Doments/train_test')