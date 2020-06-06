import os
import numpy as np
import sys
# from sklearn import F
path_mov = r'E:\Avinash\miscellaneous\project\av_segmentation\multisensory\data\crossfire.mp4'
aud = get_audio(path_mov)
path_aud = os.path.join(os.path.split(path_mov)[0], 'test.mp4')
print(path_aud)
aud.write_audiofile(path_aud)