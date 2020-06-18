import os, sys
import numpy as np
import importlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import streamlit as st

import readWriteMedia
import preProcess
import downloadMedia
from util import imgTools

import IPython.display as ipd
from nmfTools.NMFtoolbox.forwardSTFT import forwardSTFT
from nmfTools.NMFtoolbox.inverseSTFT import inverseSTFT
from nmfTools.NMFtoolbox.initTemplates import initTemplates
from nmfTools.NMFtoolbox.initActivations import initActivations
from nmfTools.NMFtoolbox.NMFD import NMFD
from nmfTools.NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from nmfTools.NMFtoolbox.visualizeComponentsNMF import visualizeComponentsNMF
from nmfTools.NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy
from scipy.io import wavfile as wav

apDir = r'/home/avi/Documents/code/python'
if apDir not in sys.path:
    sys.path.append(apDir)
from apCode.machineLearning import ml as mlearn
import apCode.SignalProcessingTools as spt

def min_to_sec(m):
    f = np.floor(m)
    d = m-f
    return f*60 + d*100

#%%
outDir = r'/home/avi/Documents/notes'
yt_url = r'https://www.youtube.com/watch?v=0omw0NBFZt8'
t_start = min_to_sec(6.00)
t_end = min_to_sec(6.10)
path_to_mov = os.path.join(outDir, 'movie.mp4')
# downloadMedia.yt_vid_to_mp4(yt_url, path_to_mov,
#                             start_time=t_start, stop_time=t_end)

aud, vid = readWriteMedia.separate_streams(path_to_mov)

#%% Get ROI mask
path_rois = os.path.join(outDir, 'RoiSet.zip')
imgDims = vid.imgs.shape[1:]
mask, _ = mlearn.readImageJRois(path_rois, imgDims)
mask[mask>0] = 255
mask = mask.astype('uint8')
img = vid.imgs[0]

# imgTools.saveImages(mask, imgDir=outDir, fmt='png',
#                     imgNames=['mask.png'])
#
# mask_img = (mask/mask.max())*img
# imgTools.saveImages(mask_img.astype('uint8'), imgDir=outDir,
#                     fmt='png', imgNames=['masked_img.png'])


#%% Estimate optical flow
# imgs = preProcess.resize_preserve_aspect_ratio(vid.imgs)
# imgs = np.gradient(imgs, axis=0)
#
# mag_flow, _ = estimate_optical_flow(imgs)
# mag_flow = (spt.standardize(mag_flow)*255).astype('uint8')
# # ang_flow = (spt.standardize(ang_flow)*255).astype(imgs.dtype)
# mov = readWriteMedia.to_video_clip(mag_flow, vid.fps)
# mov.write_videofile(os.path.join(outDir, 'movie_flow_mag.mp4'))



#%% 1. Read audio
x, fs = aud.ts, aud.fps

# make monaural if necessary
x = make_monaural(x)
x = pcmInt16ToFloat32Numpy(x)


#%% 2. Compute STFT
# spectral parameters
paramSTFT = dict()
paramSTFT['blockSize'] = 2048
paramSTFT['hopSize'] = 512
paramSTFT['winFunc'] = np.hanning(paramSTFT['blockSize'])
paramSTFT['reconstMirror'] = True
paramSTFT['appendFrame'] = True
paramSTFT['numSamples'] = len(x)

# STFT computation
X, A, P = forwardSTFT(x, paramSTFT)

# get dimensions and time and freq resolutions
numBins, numFrames = X.shape
deltaT = paramSTFT['hopSize'] / fs
deltaF = fs / paramSTFT['blockSize']


#%% 3. Apply NMF variants to STFT magnitude
# set common parameters
numComp = 3
numIter = 30
numTemplateFrames = 8

# generate initial guess for templates
paramTemplates = dict()
paramTemplates['deltaF'] = deltaF
paramTemplates['numComp'] = numComp
paramTemplates['numBins'] = numBins
paramTemplates['numTemplateFrames'] = numTemplateFrames
initW = initTemplates(paramTemplates,'drums')

# generate initial activations
paramActivations = dict()
paramActivations['numComp'] = numComp
paramActivations['numFrames'] = numFrames

initH = initActivations(paramActivations,'uniform')

# NMFD parameters
paramNMFD = dict()
paramNMFD['numComp'] = numComp
paramNMFD['numFrames'] = numFrames
paramNMFD['numIter'] = numIter
paramNMFD['numTemplateFrames'] = numTemplateFrames
paramNMFD['initW'] = initW
paramNMFD['initH'] = initH

# NMFD core method
nmfdW, nmfdH, nmfdV, divKL, _ = NMFD(A, paramNMFD)

# alpha-Wiener filtering
nmfdA, _ = alphaWienerFilter(A, nmfdV, 1.0)


#%% 4. Visualize
paramVis = dict()
paramVis['deltaT'] = deltaT
paramVis['deltaF'] = deltaF
paramVis['endeSec'] = 3.8
paramVis['fontSize'] = 14
fh1, _ = visualizeComponentsNMF(A, nmfdW, nmfdH, nmfdA, paramVis)
import matplotlib.pyplot as plt
plt.show()

#%% 5. Write audio
audios = []
# resynthesize results of NMF with soft constraints and score information
for k in range(numComp):
    Y = nmfdA[k] * np.exp(1j * P);
    y, _ = inverseSTFT(Y, paramSTFT)

    audios.append(y)
    # save result
out_filepath = os.path.join(outDir, f'drums_{k}.wav')
y_out = audios[0] + audios[2]
wav.write(filename=out_filepath, rate=fs, data=spt.standardize(y_out)*2-1)

