# from collections import namedtuple
import numpy as np
import librosa
from cv2 import resize
from dask import delayed, compute
from readWriteMedia import VideoFileReader, separate_streams, rgb2gray, AudioSignal


def preprocess_audio_signal(audio_signal, vid_slice_params):
    slice_duration_ms = vid_slice_params['slice_duration_ms']
    vid_frame_rate = vid_slice_params['frame_rate']
    n_vid_slices = vid_slice_params['n_slices']
    samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
    signal_length = samples_per_slice * n_vid_slices
    if audio_signal.get_number_of_samples() < signal_length:
        audio_signal.pad_with_zeros(signal_length)
    else:
        audio_signal.truncate(signal_length)
    n_fft = int(float(audio_signal.get_sample_rate()) / vid_frame_rate)
    hop_length = int(n_fft / 4)
    mel_spectrogram, phase = signal_to_spectrogram(audio_signal, n_fft, hop_length, mel=True, db=True)
    spectrogram_samples_per_slice = int(samples_per_slice / hop_length)
    n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)
    slices = [
        mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
        for i in range(n_slices)]
    return np.stack(slices)


def audio_video_slices_from_file(movie_path, slice_duration_ms=1000, dims=(144, 256)):
    """
    Given the path to a video file (video with audio track, e.g. .mp4 file), returns
    slices of the video, such that each slice is composed of as many frames as equivalent to
    slice_duration_ms, given the frame rate
    Parameters
    ----------
    movie_path: str
        Path to video + audio file (e.g., mp4)
    slice_duration_ms: int
        The duratio of each video slice
    dims

    Returns
    -------

    """
    print("preprocessing %s" % movie_path)
    aud, vid = separate_streams(movie_path)
    if np.ndim(vid) == 4:
        frames = rgb2gray(vid.imgs)
    frames_rs = resize_preserve_aspect_ratio(frames, dims=dims)
    n_frames = len(frames_rs)
    frames_rs = frames_rs.transpose((1, 2, 0))
    frame_rate = vid.fps
    frames_per_slice = int((float(slice_duration_ms) / 1000) * frame_rate)
    n_slices = int(float(n_frames) / frames_per_slice)
    vid_slices = [frames_rs[:, :, (i * frames_per_slice):((i + 1) * frames_per_slice)]
                  for i in range(n_slices)]
    vid_slice_params = dict(slice_duration_ms=slice_duration_ms, frame_rate=vid_frame_rate,
                            n_slices=n_slices)
    aud_signal = AudioSignal(aud.ts, aud.fps)
    # aud_slices = preprocess_audio_signal(aud_signal)
    return np.stack(vid_slices)


def resize_preserve_aspect_ratio(imgs, dims=(144, 256)):
    """
    Resize images to specified dimensions after padding to preserve aspect ratio
    Parameters
    ----------
    imgs: array, ([nImgs, ]imgHeight, imgWidth)
    dims: 2-tuple
        Final image dimensions
    Returns
    -------
    imgs_rs: array, ([nImgs, ]dims[0], dims1)
    """
    if np.ndim(imgs)==2:
        imgs = imgs[None,...]
    width = imgs.shape[2]
    height = imgs.shape[1]
    ar = round(dims[1]/dims[0], 2)
    ar_imgs = round(width/height, 2)
    # print(f'AR = {ar_imgs}, Desired AR = {ar}')
    if ar_imgs < ar:
        width = int(width*ar/ar_imgs)
        pad = width-imgs.shape[2]
        first = pad//2
        second = pad - first
        imgs_pad = np.pad(imgs,((0, 0), (0, 0), (first, second)), mode='linear_ramp')
    elif ar_imgs > ar:
        height = int(height*ar_imgs/ar)
        pad = height-imgs.shape[1]
        first = pad//2
        second = pad - first
        imgs_pad = np.pad(imgs, ((0, 0), (first, second), (0, 0)))
    else:
        imgs_pad = imgs
    # print(imgs_pad.shape[1:])
    imgs_rs = [delayed(resize)(img, (dims[1], dims[0])) for img in imgs_pad]
    imgs_rs = np.array(compute(*imgs_rs))
    return np.squeeze(imgs_rs)


def signal_to_spectrogram(audio_signal, n_fft, hop_length, mel=True, db=True):
    signal = audio_signal.get_data(channel_index=0)
    D = librosa.core.stft(signal.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.core.magphase(D)
    if mel:
        mel_filterbank = librosa.filters.mel(sr=audio_signal.get_sample_rate(),
                                             n_fft=n_fft, n_mels=80, fmin=0,
                                             fmax=8000)
        magnitude = np.dot(mel_filterbank, magnitude)
    if db:
        magnitude = librosa.amplitude_to_db(magnitude)
    return magnitude, phase