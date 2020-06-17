# from collections import namedtuple
import numpy as np
import librosa
from cv2 import resize
from dask import delayed, compute
from readWriteMedia import VideoFileReader, separate_streams, rgb2gray, AudioSignal, AudioMixer


def preprocess_audio_signal(audio_signal, vid_slice_params):
    """
    Given the audio_signal object along with corresponding video slice parameters,
    returns spectrogam slices matching video slices. See audio_video_slices_from_file
    Parameters
    ----------
    audio_signal: object
        Audio signal object obtained as audio_signal = AudioSignal(audio_timeseries, sampling_rate)
    vid_slice_params: dict
        Parameters pertaining to video slices that are used to extrac audio slices

    Returns
    -------
    aud_slices: array, (nSlices, *spectrogramDimensions)
    """
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
    aud_slices: array, (nSlices, *spectrogramDims, nSamplesPerSlice)
    vid_slices: array, (nSlices, *imgDims_resized, nSamplesPerSlice)
    """
    print("preprocessing %s" % movie_path)
    aud, vid = separate_streams(movie_path)
    if np.ndim(vid.imgs) == 4:
        frames = rgb2gray(vid.imgs)
    else:
        frames = vid.imgs
    frames_rs = resize_preserve_aspect_ratio(frames, dims=dims)
    n_frames = len(frames_rs)
    frames_rs = frames_rs.transpose((1, 2, 0))
    frame_rate = vid.fps
    frames_per_slice = int((float(slice_duration_ms) / 1000) * frame_rate)
    n_slices = int(float(n_frames) / frames_per_slice)
    vid_slices = [frames_rs[:, :, (i * frames_per_slice):((i + 1) * frames_per_slice)]
                  for i in range(n_slices)]
    vid_slice_params = dict(slice_duration_ms=slice_duration_ms, frame_rate=frame_rate,
                            n_slices=n_slices)
    aud_signal = AudioSignal(aud.ts, aud.fps)
    aud_slices = preprocess_audio_signal(aud_signal, vid_slice_params)
    vid_slices, aud_slices = map(np.stack, (vid_slices, aud_slices))
    n_slices = min(aud_slices.shape[0], vid_slices.shape[0])
    vid_slices = vid_slices[:n_slices]
    aud_slices = aud_slices[:n_slices]
    out = dict(vid_slices=vid_slices, vid_slice_params=vid_slice_params,
               aud_signal=aud_signal, aud_slices=aud_slices)

    return out


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


def reconstruct_signal_from_spectrogram(magnitude, phase, sample_rate, n_fft,
                                        hop_length, mel=True, db=True):
    """
    Inverse function for signal_to_spectrogram
    Parameters
    ----------
    magnitude: array, (n_mels, )
        Spectrogram magnitude
    phase: array, (n_mels, )
        Spectrogram phase
    sample_rate: int
        Audio sample rate
    n_fft: int
        Number of Fourier scales
    hop_length
    mel
    db

    Returns
    -------

    """
    if db:
        magnitude = librosa.db_to_amplitude(magnitude)
    if mel:
        mel_filterbank = librosa.filters.mel(sr=sample_rate, n_fft=n_fft,
                                             n_mels=80, fmin=0, fmax=8000)
        magnitude = np.dot(np.linalg.pinv(mel_filterbank), magnitude)
    signal = librosa.istft(magnitude * phase, hop_length=hop_length)
    return AudioSignal(signal, sample_rate)


def reconstruct_speech_signal(mixed_signal, speech_spectrograms, video_frame_rate):
    n_fft = int(float(mixed_signal.get_sample_rate()) / video_frame_rate)
    hop_length = int(n_fft / 4)
    _, original_phase = signal_to_spectrogram(mixed_signal, n_fft, hop_length, mel=True, db=True)
    speech_spectrogram = np.concatenate(list(speech_spectrograms), axis=1)
    spectrogram_length = min(speech_spectrogram.shape[1], original_phase.shape[1])
    speech_spectrogram = speech_spectrogram[:, :spectrogram_length]
    original_phase = original_phase[:, :spectrogram_length]
    return reconstruct_signal_from_spectrogram(speech_spectrogram, original_phase,
                                               mixed_signal.get_sample_rate(), n_fft,
                                               hop_length, mel=True, db=True)


# def preprocess_audio_pair(speech_file_path, noise_file_path, slice_duration_ms,
#                                n_video_slices, video_frame_rate):
#     print("preprocessing pair: %s, %s" % (speech_file_path, noise_file_path))
#     speech_signal = AudioSignal.from_wav_file(speech_file_path)
#     noise_signal = AudioSignal.from_wav_file(noise_file_path)
#     while noise_signal.get_number_of_samples() < speech_signal.get_number_of_samples():
#         noise_signal = AudioSignal.concat([noise_signal, noise_signal])
#
#     noise_signal.truncate(speech_signal.get_number_of_samples())
#     factor = AudioMixer.snr_factor(speech_signal, noise_signal, snr_db=0)
#     noise_signal.amplify_by_factor(factor)
#     mixed_signal = AudioMixer.mix([speech_signal, noise_signal], mixing_weights=[1, 1])
#     mixed_spectrograms = preprocess_audio_signal(mixed_signal, slice_duration_ms, n_video_slices, video_frame_rate)
#     speech_spectrograms = preprocess_audio_signal(speech_signal, slice_duration_ms, n_video_slices, video_frame_rate)
#     noise_spectrograms = preprocess_audio_signal(noise_signal, slice_duration_ms, n_video_slices, video_frame_rate)
#     return mixed_spectrograms, speech_spectrograms, noise_spectrograms, mixed_signal


def preprocess_video_pair(speech_file_path, noise_file_path, slice_duration_ms=1000,
                          mixing_weights=(1, 1), snr_db=10):
    """
    When given the paths to the file with sound of interest and file with noise
    returns a dictionary wtith information pertinent to training
    Parameters
    ----------
    speech_file_path: str
        Path to video file with sound of interest
    noise_file_path: str
        Path to video file with noise
    slice_duration_ms: int
        Duration of slices for training
    mixing_weights: 2-tuple or list
    snr_db: scalar
        Determines the amplification of noise. High values lead to less noise amplification
    Returns
    -------

    """
    print("preprocessing pair: %s, %s" % (speech_file_path, noise_file_path))
    speech_dic = audio_video_slices_from_file(speech_file_path, slice_duration_ms)
    speech_signal = speech_dic['aud_signal']
    vid_slices = speech_dic['vid_slices']
    noise_dic = audio_video_slices_from_file(noise_file_path, slice_duration_ms)
    noise_signal = noise_dic['aud_signal']
    while noise_signal.get_number_of_samples() < speech_signal.get_number_of_samples():
        noise_signal = AudioSignal.concat([noise_signal, noise_signal])

    noise_signal.truncate(speech_signal.get_number_of_samples())
    factor = AudioMixer.snr_factor(speech_signal, noise_signal, snr_db=10)
    noise_signal.amplify_by_factor(factor)
    mixed_signal = AudioMixer.mix([speech_signal, noise_signal],
                                  mixing_weights=list(mixing_weights))
    mixed_spectrograms = preprocess_audio_signal(mixed_signal, speech_dic['vid_slice_params'])
    speech_spectrograms = preprocess_audio_signal(speech_signal, speech_dic['vid_slice_params'])
    noise_spectrograms = preprocess_audio_signal(noise_signal, speech_dic['vid_slice_params'])
    out = dict(mixed_spectrograms=mixed_spectrograms,
               speech_spectrograms=speech_spectrograms,
               noise_spectrograms=noise_spectrograms,
               speech_signal=speech_signal,
               noise_signal=noise_signal,
               mixed_signal=mixed_signal,
               vid_slices=vid_slices)
    return out


class VideoNormalizer(object):
    def __init__(self, video_samples):
        # video_samples: slices x height x width x frames_per_slice
        self.__mean_image = np.mean(video_samples, axis=(0, 3))
        self.__std_image = np.std(video_samples, axis=(0, 3))

    def normalize(self, video_samples):
        for slc in range(video_samples.shape[0]):
            for frame in range(video_samples.shape[3]):
                video_samples[slc, :, :, frame] -= self.__mean_image
                video_samples[slc, :, :, frame] /= self.__std_image