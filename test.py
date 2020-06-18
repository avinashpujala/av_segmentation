import preProcess
import readWriteMedia
from networks.asfv import NeuralNetwork as nn
model = nn.__init__()
import time
import importlib
import os
import glob

importlib.reload(preProcess)
importlib.reload(readWriteMedia)

dir_ebs = r'/home/ubuntu/avinash/vol_ebs'

print(os.listdir(dir_ebs))



