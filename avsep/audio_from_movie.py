import sys
from moviepy import editor as med
import os

def get_audio_video(path_to_mov):
    vid = med.VideoFileClip(path_to_mov)
    aud = vid.audio
    return aud, vid


# path_mov = r'E:\Avinash\miscellaneous\project\av_segmentation\multisensory\data\crossfire.mp4'
path_mov = r'E:\Avinash\miscellaneous\project\av_segmentation\multisensory\data\llt.mp4'
aud, vid = get_audio_video(path_mov)
path_aud = os.path.join(os.path.split(path_mov)[0], 'aud_sub.wav')
path_vid = os.path.join(os.path.split(path_mov)[0], 'vid_sub/img%04d.png')
os.makedirs(path_vid, exist_ok=True)
print(path_aud)
vid_sub = vid.subclip(240, 270)
aud_sub = vid_sub.audio.write_audiofile(path_aud)
vid_sub.write_images_sequence(path_vid)
# aud.write_audiofile(path_aud, fps=1000)

from ffmpeg import audio, video
# video.separate_audio(path_mov, pa)