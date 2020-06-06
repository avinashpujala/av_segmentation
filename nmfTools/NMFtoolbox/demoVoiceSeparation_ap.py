import os
import numpy as np
import scipy.io.wavfile as wav
import IPython.display as ipd
import sys
codeDir =r'E:\Avinash\miscellaneous\project\av_segmentation\nmfTools'
if codeDir not in sys.path:
    sys.path.append(codeDir)
import NMFtoolbox

from NMFtoolbox.forwardSTFT import forwardSTFT
from NMFtoolbox.inverseSTFT import inverseSTFT
from NMFtoolbox.initTemplates import initTemplates
from NMFtoolbox.initActivations import initActivations
from NMFtoolbox.NMFD import NMFD
from NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from NMFtoolbox.visualizeComponentsNMF import visualizeComponentsNMF
from nmfTools.utils import make_monaural, pcmInt16ToFloat32Numpy

#%% Directories
inPath = r'E:\Avinash\miscellaneous\project\av_segmentation\multisensory\data'
outPath = inPath
filename = 'aud_sub.wav'

#%% 1. Read audio
path_aud = os.path.join(inPath, filename)
fs, x = wav.read(path_aud)

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


#%%  Apply NMF variants to STFT magnitude
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

#visualize
paramVis = dict()
paramVis['deltaT'] = deltaT
paramVis['deltaF'] = deltaF
paramVis['endeSec'] = 3.8
paramVis['fontSize'] = 14
fh1, _ = visualizeComponentsNMF(A, nmfdW, nmfdH, nmfdA, paramVis)

audios = []
# resynthesize results of NMF with soft constraints and score information
for k in range(numComp):
    Y = nmfdA[k] * np.exp(1j * P);
    y, _ = inverseSTFT(Y, paramSTFT)

    audios.append(y)
    # save result
    out_filepath = os.path.join(outPath,
                                'aud_sub_NMFD_component_{}.wav'.format(k, filename))

    wav.write(filename=out_filepath, rate=fs, data=y)