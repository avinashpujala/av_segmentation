import readWriteMedia
import downloadMedia
import os, sys
import numpy as np

import os
import numpy as np
import scipy.io.wavfile as wav
import IPython.display as ipd
import moviepy.editor as med

from nmfTools.NMFtoolbox.forwardSTFT import forwardSTFT
from nmfTools.NMFtoolbox.inverseSTFT import inverseSTFT
from nmfTools.NMFtoolbox.initTemplates import initTemplates
from nmfTools.NMFtoolbox.initActivations import initActivations
from nmfTools.NMFtoolbox.NMFD import NMFD
from nmfTools.NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from nmfTools.NMFtoolbox.visualizeComponentsNMF import visualizeComponentsNMF
from nmfTools.NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy


apDir = r'/home/avi/Documents/code/python'
if apDir not in sys.path:
    sys.path.append(apDir)
from apCode.machineLearning import ml as mlearn
# import machineLearn.ml as mlearn

def min_to_sec(m):
    f = np.floor(m)
    d = m-f
    return f*60 + d*100


yt_url = r'https://www.youtube.com/watch?v=0omw0NBFZt8'
t_start = min_to_sec(6.00)
t_end = min_to_sec(6.10)
outDir = r'/home/avi/Documents/notes'
path_to_mov = os.path.join(outDir, 'movie.mp4')
path_rois = os.path.join(outDir, 'RoiSet.zip')

# downloadMedia.yt_vid_to_mp4(yt_url, path_to_mov,
#                             start_time=t_start, stop_time=t_end)

aud, vid = readWriteMedia.separate_streams(path_to_mov)

#%% Get ROI mask
imgDims = vid.imgs.shape[1:]
mask, _ = mlearn.readImageJRois(path_rois, imgDims)



