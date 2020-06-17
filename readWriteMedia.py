from moviepy import editor as med
import numpy as np
import cv2
from scipy.io import wavfile
import imageio
import subprocess

# import sys
# dir_med = r'/home/avi/Documents/code/projects/mediaio'
# if dir_med not in sys.path:
#     sys.path.append(dir_med)
# from mediaio import audio_io, video_io #noqa


class AudioSignal:
    def __init__(self, data, sample_rate):
        self._data = np.copy(data)
        self._sample_rate = sample_rate

    @staticmethod
    def from_wav_file(wave_file_path):
        sample_rate, data = wavfile.read(wave_file_path)
        return AudioSignal(data, sample_rate)

    @staticmethod
    def from_mp4(mp4_path):
        aud, _ = separate_streams(mp4_path)
        return AudioSignal(aud.ts, aud.fps)

    def save_to_wav_file(self, wave_file_path, sample_type=np.int16):
        self.set_sample_type(sample_type)
        wavfile.write(wave_file_path, self._sample_rate, self._data)

    def get_data(self, channel_index=None):
        # data shape: (n_samples) or (n_samples, n_channels)
        if channel_index is None:
            return self._data

        if channel_index not in range(self.get_number_of_channels()):
            raise IndexError("invalid channel index")

        if channel_index == 0 and self.get_number_of_channels() == 1:
            return self._data
        return self._data[:, channel_index]

    def get_number_of_samples(self):
        return self._data.shape[0]

    def get_number_of_channels(self):
        # data shape: (n_samples) or (n_samples, n_channels)
        if len(self._data.shape) == 1:
            return 1
        return self._data.shape[1]

    def get_sample_rate(self):
        return self._sample_rate

    def get_sample_type(self):
        return self._data.dtype

    def get_format(self):
        return dict(n_channels=self.get_number_of_channels(),
                    sample_rate=self.get_sample_rate())

    def get_length_in_seconds(self):
        return float(self.get_number_of_samples()) / self.get_sample_rate()

    def set_sample_type(self, sample_type):
        sample_type_info = np.iinfo(sample_type)
        self._data = self._data.clip(sample_type_info.min, sample_type_info.max).astype(sample_type)

    def amplify(self, reference_signal):
        factor = float(np.abs(reference_signal.get_data()).max()) / np.abs(self._data).max()

        new_max_value = self._data.max() * factor
        new_min_value = self._data.min() * factor

        sample_type_info = np.iinfo(self.get_sample_type())
        if new_max_value > sample_type_info.max or new_min_value < sample_type_info.min:
            raise Exception("amplified signal exceeds audio format boundaries")

        self._data = (self._data.astype(np.float64) * factor).astype(self.get_sample_type())

    def amplify_by_factor(self, factor):
        self._data = self._data.astype(np.float64)
        self._data *= factor

    def peak_normalize(self, peak=None):
        self._data = self._data.astype(np.float64)
        if peak is None:
            peak = np.abs(self._data).max()
        self._data /= peak
        return peak

    def split(self, n_slices):
        return [AudioSignal(s, self._sample_rate) for s in np.split(self._data, n_slices)]

    def slice(self, start_sample_index, end_sample_index):
        return AudioSignal(self._data[start_sample_index:end_sample_index], self._sample_rate)

    def pad_with_zeros(self, new_length):
        if self.get_number_of_samples() > new_length:
            raise Exception("cannot zero-pad for shorter signal length")
        new_shape = list(self._data.shape)
        new_shape[0] = new_length
        self._data = np.copy(self._data)
        self._data.resize(new_shape)

    def truncate(self, new_length):
        if self.get_number_of_samples() < new_length:
            raise Exception("cannot truncate for longer signal length")
        self._data = self._data[:new_length]

    @staticmethod
    def concat(signals):
        for signal in signals:
            if signal.get_format() != signals[0].get_format():
                raise Exception("concating audio signals with different formats is not supported")
        data = [signal.get_data() for signal in signals]
        return AudioSignal(np.concatenate(data), signals[0].get_sample_rate())


class AudioMixer:
    @staticmethod
    def mix(audio_signals, mixing_weights=None):
        if mixing_weights is None:
            mixing_weights = [1] * len(audio_signals)
        reference_signal = audio_signals[0]
        mixed_data = np.zeros(shape=reference_signal.get_data().shape, dtype=np.float64)
        for i, signal in enumerate(audio_signals):
            if signal.get_format() != reference_signal.get_format():
                raise Exception("mixing audio signals with different format is not supported")
            mixed_data += (float(mixing_weights[i])) * signal.get_data()
        return AudioSignal(mixed_data, reference_signal.get_sample_rate())

    @staticmethod
    def snr_factor(signal, noise, snr_db):
        s = signal.get_data()
        n = noise.get_data()

        if s.size != n.size:
            raise Exception('signal and noise must have the same length')
        eq = np.sqrt(np.var(s) / np.var(n))
        factor = eq * (10 ** (-snr_db / 20.0))
        return factor


class FFMPEG:
    @staticmethod
    def downsample(input_audio_file_path, output_audio_file_path, sample_rate):
        subprocess.check_call(["ffmpeg", "-i", input_audio_file_path, "-ar",
                               str(sample_rate), output_audio_file_path, "-y"])
    @staticmethod
    def merge(input_video_file_path, input_audio_file_path, output_video_file_path):
        subprocess.check_call(["ffmpeg", '-hide_banner', '-loglevel', 'panic', "-i",
                               input_video_file_path, "-i", input_audio_file_path,
                               "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                               output_video_file_path])


class VidImgs:
    def __init__(self, vid):
        self.imgs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                              for img in vid.iter_frames()])
        self.fps = vid.fps
        self.dur = vid.duration

    def to_video_clip(self):
        imgs = gray2rgb(self.imgs)
        clips = [med.ImageClip(img).set_duration(1 / self.fps) for img in imgs]
        mov = med.concatenate_videoclips(clips, method='compose')
        mov = mov.set_fps(self.fps)
        return mov

    def write_videofile(self, fname='movie.mp4', *args, **kwargs):
        self.to_videoClip().write_videofile(fname, self.fps, *args, **kwargs)


class AudTs:
    def __init__(self, aud):
        self.ts = aud.to_soundarray().mean(axis=1)
        self.fps = aud.fps
        self.dur = aud.duration


class VideoFileReader:
    def __init__(self, video_file_path):
        self._video_fd = imageio.get_reader(video_file_path)

    def close(self):
        self._video_fd.close()

    def read_all_frames(self, convert_to_gray_scale=False):
        # if convert_to_gray_scale:
        #     video_shape = (self.get_frame_count(), self.get_frame_height(), self.get_frame_width())
        # else:
        #     video_shape = (self.get_frame_count(), self.get_frame_height(), self.get_frame_width(), 3)

        # frames = np.ndarray(shape=video_shape, dtype=np.uint8)
        mov = []
        for i in range(self.get_frame_count()):
            # frames[i, ] = self.read_next_frame(convert_to_gray_scale=convert_to_gray_scale)
            mov.append(self._video_fd.get_data(i))
        mov = np.array(mov, dtype=np.uint8)
        if convert_to_gray_scale:
            mov = rgb2gray(mov)
        return mov

    def read_next_frame(self, convert_to_gray_scale=False):
        frame = self._video_fd.get_next_data()

        if convert_to_gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def get_frame_rate(self):
        return self._video_fd.get_meta_data()["fps"]

    def get_frame_size(self):
        return self._video_fd.get_meta_data()["size"]

    def get_frame_count(self):
        # return self._video_fd.get_length()
        return self._video_fd.count_frames()

    def get_frame_width(self):
        return self.get_frame_size()[0]

    def get_frame_height(self):
        return self.get_frame_size()[1]

    def get_format(self):
        return dict(frame_rate=self.get_frame_rate(),
                    frame_width=self.get_frame_width(),
                    frame_height=self.get_frame_height())

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class VideoFileWriter:
    def __init__(self, video_file_path, frame_rate):
        self._video_fd = imageio.get_writer(video_file_path, fps=frame_rate)

    def close(self):
        self._video_fd.close()

    def write_frame(self, frame):
        self._video_fd.append_data(frame)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


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
        a = AudTs(aud)
        v = VidImgs(vid)
    return a, v


def to_video_clip(imgs, fps):
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
