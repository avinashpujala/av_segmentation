import glob
from os.path import join
import processMedia
import numpy as np
import importlib
importlib.reload(processMedia)
import matplotlib.pyplot as plt

idx = 0
dir_vids = r'C:\Users\Avi\Google Drive\Code\projects\av_segmentation\data\avspeech'
filePaths = glob.glob(join(dir_vids, '*.mp4'))
fp = filePaths[idx]
# fp = join(dir_vids, '_-_1K9uCid4_270.067000_275.767000.mp4')
aud, vid = processMedia.separate_streams(fp)
foo = np.gradient(vid.imgs[:, ::2, ::2], axis=0).sum(axis=-1).sum(axis=-1)

#%%
t_aud = np.linspace(0, aud.dur, len(aud.ts))
t_vid = np.linspace(0, vid.dur, vid.imgs.shape[0])
plt.plot(t_aud, aud.ts/aud.ts.max())
plt.plot(t_vid, foo/foo.max())

"""
# yt_url = r'https://www.youtube.com/watch?v=gMz4r5RLvKs'
# yt_url = r'https://www.youtube.com/watch?v=fregObNcHC8'
# yt_url = yt_url.split('?v=')[-1]
"""