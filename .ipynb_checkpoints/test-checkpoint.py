import glob
from os.path import join
import readWriteMedia
import numpy as np
import importlib
importlib.reload(readWriteMedia)
import matplotlib.pyplot as plt

idx = 0
dir_vids = r'C:\Users\Avi\Google Drive\Code\projects\av_segmentation\data\avspeech'
filePaths = glob.glob(join(dir_vids, '*.mp4'))
fp = filePaths[idx]
# fp = join(dir_vids, '_-_1K9uCid4_270.067000_275.767000.mp4')
aud, vid = readWriteMedia.separate_streams(fp)
imgs_sub = vid.imgs[:, ::2, ::2]
dImgs = np.gradient(imgs_sub, axis=0).astype(imgs_sub.dtype)
dImgs2 = np.gradient(imgs_sub-imgs_sub.mean(), axis=0).astype(imgs_sub.dtype)
foo = dImgs.mean(axis=-1).mean(axis=-1)
foo2 = dImgs2.mean(axis=-1).mean(axis=-1)

#%%
t_aud = np.linspace(0, aud.dur, len(aud.ts))
t_vid = np.linspace(0, vid.dur, vid.imgs.shape[0])
plt.plot(t_aud, aud.ts/aud.ts.max(), label='Audio')
plt.plot(t_vid, foo/foo.max(), label='Diff only')
plt.plot(t_vid, foo2/foo2.max(), ls ='--', label='Mean sub + diff')
plt.legend()

#%%
# processMedia.to_videoClip(dImgs, vid.fps).write_videofile('dMovie.mp4')
# processMedia.to_videoClip(dImgs2, vid.fps).write_videofile('dMovie2.mp4')


"""
# yt_url = r'https://www.youtube.com/watch?v=gMz4r5RLvKs'
# yt_url = r'https://www.youtube.com/watch?v=fregObNcHC8'
# yt_url = yt_url.split('?v=')[-1]
"""