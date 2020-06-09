# from mediaio import audio_io, video_io
from moviepy import editor as med
import numpy as np
import cv2


class AudClass(object):
    def __init__(self, aud):
        self.ts = aud.to_soundarray().mean(axis=1)
        self.fps = aud.fps
        self.dur = aud.duration


class VidClass:
    def __init__(self, vid):
        self.imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                              for img in vid.iter_frames()])
        self.fps = vid.fps
        self.dur = vid.duration

    def to_videoClip(self):
        # imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in self.imgs]
        imgs = gray2rgb(self.imgs)
        clips = [med.ImageClip(img).set_duration(1 / self.fps) for img in imgs]
        mov = med.concatenate_videoclips(clips, method='compose')
        return mov

    def write_videofile(self, fname='movie.mp4', *args, **kwargs):
        self.to_videoClip().write_videofile(fname, self.fps, *args, **kwargs)


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
        a = AudClass(aud)
        v = VidClass(vid)
    return a, v


def to_videoClip(imgs, fps):
    imgs = gray2rgb(imgs)
    clips = [med.ImageClip(img).set_duration(1 / fps) for img in imgs]
    mov = med.concatenate_videoclips(clips, method='compose')
    mov = mov.set_fps(fps)
    return mov


def gray2rgb(imgs):
    imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in imgs]
    return np.array(imgs)


def rgb2gray(imgs):
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]
    return np.array(imgs)
