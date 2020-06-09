# from mediaio import audio_io, video_io
from moviepy import editor as med
import numpy as np
import cv2

class __get_aud(object):
    def __init__(self, aud):
        self.ts = aud.to_soundarray().mean(axis=1)
        self.fps = aud.fps
        self.dur = aud.duration


class __get_vid:
    def __init__(self, vid):
        self.imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                              for img in vid.iter_frames()])
        self.fps = vid.fps
        self.dur = vid.duration

    def to_videoClip(self):
        clips = [med.ImageClip(img).set_duration(1/self.fps) for img in self.imgs]
        mov = med.concatenate_videoclips(clips, method='compose')
        return mov


def separate_streams(path_to_mov):
    """
    Separate audio and video streams from media (e.g. mp4)
    Parameters
    ----------
    path_to_mov: str
        Path to audio-video (e.g., mp4)
    Returns
    -------
    aud: audio stream object (moviepy class)
    vid: video stream object (moviepy class)
    """
    with med.VideoFileClip(path_to_mov) as vid:
        aud = vid.audio
        a = __get_aud(aud)
        v = __get_vid(vid)
    return a, v