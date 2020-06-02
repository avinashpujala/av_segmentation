import sys
import os
from multiprocessing.pool import ThreadPool
import youtube_dl
import ffmpeg
import numpy as np
from util.fileTools import get_files


# class VidInfo:
#     def __init__(self, yt_id, start_time, end_time, outDir):
#         self.yt_id = yt_id
#         self.start_time = float(start_time)
#         self.end_time = float(end_time)
#         self.out_filename = os.path.join(outDir,
#                                          yt_id + '_' + start_time +
#                                          '_' + end_time + '.mp4')

def download_av_speech(path_to_csv, out_dir='train', n_files=270000):
    """
    Downloads AVSpeech Dataset
    https://looking-to-listen.github.io/avspeech/download.html

    Adapted from code by Nabarun Goswami
    https://github.com/naba89/AVSpeechDownloader
    Parameters
    ----------
    path_to_csv: str
        Path to csv file containing data locations
    out_dir: str
        Name of the subdirectory within path_to_csv where
        videos (mp4 files) will be downloaded
    n_files: int
        Since this is a very large dataset (~ 2.2TB), only
        these many files will be randomly saved.
    Returns
    -------
    out_dir: str
        Absolute path of the directory with downloaded videos
    """
    class VidInfo:
        def __init__(self, yt_id, start_time, end_time, outDir):
            self.yt_id = yt_id
            self.start_time = float(start_time)
            self.end_time = float(end_time)
            self.out_filename = os.path.join(outDir,
                                             yt_id + '_' + start_time +
                                             '_' + end_time + '.mp4')
    if not os.path.isdir(out_dir):
        out_dir = os.path.join(os.path.split(path_to_csv)[0], out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fNames_pre = get_files(out_dir)

    with open(path_to_csv, 'r') as f:
        lines = f.readlines()
        inds = np.random.randint(0, len(lines), n_files)
        lines = [lines[ind].split(',') for ind in inds]
        fNames = [x[0] + '_' + x[1] + '_' + x[2] + '.mp4' for x in lines]
        inds = np.nonzero(~np.isin(fNames, fNames_pre))[0]
        lines = [lines[ind] for ind in inds]
        vidinfos = [VidInfo(x[0], x[1], x[2], out_dir) for x in lines]

    bad_files = open(os.path.join(out_dir, 'bad_files.txt'), 'w')
    results = ThreadPool(5).imap_unordered(__download, vidinfos)
    cnt = 0
    for r in results:
        cnt += 1
        print(cnt, '/', len(vidinfos), r)
        if 'ERROR' in r:
            bad_files.write(r + '\n')
    bad_files.close()
    return out_dir


def __download(vidinfo):
    yt_base_url = 'https://www.youtube.com/watch?v='
    yt_url = yt_base_url+vidinfo.yt_id
    ydl_opts = {'format': '22/18',
                'quiet': True,
                'ignoreerrors': True,
                'no_warnings': True}
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            download_url = ydl.extract_info(url=yt_url, download=False)['url']
    except Exception:
        return_msg = '{}, ERROR (youtube)!'.format(vidinfo.yt_id)
        return return_msg
    try:
        (
            ffmpeg
                .input(download_url, ss=vidinfo.start_time, to=vidinfo.end_time)
                .output(vidinfo.out_filename, format='mp4', r=25, vcodec='libx264',
                        crf=18, preset='veryfast', pix_fmt='yuv420p', acodec='aac',
                        audio_bitrate=128000, strict='experimental')
                .global_args('-y')
                .global_args('-loglevel', 'error')
                .run()
        )
    except Exception:
        return_msg = '{}, ERROR (ffmpeg)!'.format(vidinfo.yt_id)
        return return_msg
    return '{}, DONE!'.format(vidinfo.yt_id)