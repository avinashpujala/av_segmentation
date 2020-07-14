import numpy as np
import librosa
from cv2 import resize
import os
import glob
import pickle
import h5py
from dask import delayed, compute

from readWriteMedia import VideoFileReader, separate_streams, rgb2gray, AudioSignal, AudioMixer
from util import util
from util import fileTools as ft


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
    mel_spectrogram, phase = signal_to_spectrogram(audio_signal, n_fft, hop_length,
                                                   mel=True, db=True)
    spectrogram_samples_per_slice = int(samples_per_slice / hop_length)
    n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)
    slices = [
        mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
        for i in range(n_slices)]
    return np.stack(slices)


def audio_video_slices_from_file(movie_path, slice_duration_ms=400,
                                 dims=(144, 256), verbose: bool = False):
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
    dims: tuple
        Image dimensions after resizing
    verbose: bool
        If True, prints path to movie file
    Returns
    -------
    aud_slices: array, (nSlices, *spectrogramDims, nSamplesPerSlice)
    vid_slices: array, (nSlices, *imgDims_resized, nSamplesPerSlice)
    """
    if verbose:
        print("preprocessing %s" % movie_path)
    try:
        if os.path.exists(movie_path):
            aud, vid = separate_streams(movie_path)
        else:
            return None
    except Exception as e:
        print(e)
        return None
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


def get_file_paths(sound_dir, noise_dir=None, n_files=None, ext='mp4'):
    if noise_dir is None:
        noise_dir = sound_dir
    ext = ext.split('.')[-1]
    sound_paths = glob.glob(os.path.join(sound_dir, f'*.{ext}'))
    if n_files is None:
        n_files = len(sound_paths)
    print(f'{n_files} file paths')
    noise_paths = glob.glob(os.path.join(noise_dir, f'*.{ext}'))
    noise_paths = np.union1d(noise_paths, sound_paths)
    np.random.shuffle(noise_paths)
    np.random.shuffle(sound_paths)
    noise_paths = noise_paths[:len(sound_paths)]
    return sound_paths[:n_files], noise_paths[:n_files]


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
    imgs_rs_delayed = [delayed(resize)(img, (dims[1], dims[0])) for img in imgs_pad]
    imgs_rs = np.array(compute(*imgs_rs_delayed))
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
    _, original_phase = signal_to_spectrogram(mixed_signal, n_fft, hop_length,
                                              mel=True, db=True)
    speech_spectrogram = np.concatenate(list(speech_spectrograms), axis=1)
    spectrogram_length = min(speech_spectrogram.shape[1], original_phase.shape[1])
    speech_spectrogram = speech_spectrogram[:, :spectrogram_length]
    original_phase = original_phase[:, :spectrogram_length]
    return reconstruct_signal_from_spectrogram(speech_spectrogram, original_phase,
                                               mixed_signal.get_sample_rate(), n_fft,
                                               hop_length, mel=True, db=True)


def preprocess_video_pair(speech_file_path, noise_file_path, slice_duration_ms=400,
                          mixing_weights=(1, 1), snr_db=0, relevant_only: bool = True,
                          verbose: bool = False):
    """
    When given the paths to the file with sound of interest and file with noise
    returns a dictionary wtith information pertinent to training
    Parameters
    ----------
    speech_file_path: str or iterable of str
        Path to video file with sound of interest
    noise_file_path: str or iterable of str
        Path to video file with noise
    slice_duration_ms: int
        Duration of slices for training
    mixing_weights: 2-tuple or list
    snr_db: scalar
        Determines the amplification of noise. High values lead to less noise amplification
    relevant_only: bool
        If True, then only return variables that can be directly used for training the network, i.e.,
        (mixed_spectrograms, video_samples, speech_spectrogram)
    verbose: bool
        If True, prints file paths
    Returns
    -------
    out: dict
        Key-value pairs:
        vid_slices: array, (nSices, *imgDims, nFrames)
            Video slices with # of frames determined by parameter slice_duration_ms
        speech_spectrograms: array, (nSlices, nFreqBins, nTimeBins)
            Sound spectrograms from video at file speech_file_path
        noise_spectrograms: array, (nSlices, nFreqBins, nTimeBins)
            Noise spectrograms from video at file noise_file_path
        mixed_spectrograms: Mixture of speech and noise spectrograms
        speech_signal: AudioSignal object
            Contains speech timeseries and useful methods
        noise_signal, mixed_signal: Self-evident
    """
    if isinstance(speech_file_path, str):
        speech_dic = audio_video_slices_from_file(speech_file_path, slice_duration_ms)
        if speech_dic is None:
            return None
        speech_signal = speech_dic['aud_signal']
        vid_slices = speech_dic['vid_slices']
        noise_dic = audio_video_slices_from_file(noise_file_path, slice_duration_ms,
                                                 verbose=verbose)
        if noise_dic is None:
            return None
        noise_signal = noise_dic['aud_signal']
        while noise_signal.get_number_of_samples() < speech_signal.get_number_of_samples():
            noise_signal = AudioSignal.concat([noise_signal, noise_signal])

        try:
            noise_signal.truncate(speech_signal.get_number_of_samples())
            factor = AudioMixer.snr_factor(speech_signal, noise_signal, snr_db=snr_db)
            noise_signal.amplify_by_factor(factor)
            mixed_signal = AudioMixer.mix([speech_signal, noise_signal],
                                          mixing_weights=list(mixing_weights))
            mixed_spectrograms = preprocess_audio_signal(mixed_signal,
                                                         speech_dic['vid_slice_params'])
            speech_spectrograms = preprocess_audio_signal(speech_signal,
                                                          speech_dic['vid_slice_params'])
            noise_spectrograms = preprocess_audio_signal(noise_signal,
                                                         speech_dic['vid_slice_params'])
        except Exception as e:
            print(e)
            return None

        if relevant_only:
            out = mixed_spectrograms, vid_slices, speech_spectrograms
        else:
            out = dict(mixed_spectrograms=mixed_spectrograms,
                       speech_spectrograms=speech_spectrograms,
                       noise_spectrograms=noise_spectrograms,
                       speech_signal=speech_signal,
                       noise_signal=noise_signal,
                       mixed_signal=mixed_signal,
                       vid_slices=vid_slices)
    else:
        try:
            out = [delayed(preprocess_video_pair)(sfp, nfp, slice_duration_ms=slice_duration_ms,
                                                  mixing_weights=mixing_weights,
                                                  snr_db=snr_db, verbose=verbose)
                   for sfp, nfp in zip(speech_file_path, noise_file_path)]
            out = compute(*out)
        except Exception as e:
            print(e)
            out = [preprocess_video_pair(sfp, nfp, slice_duration_ms=slice_duration_ms,
                                         mixing_weights=mixing_weights, snr_db=snr_db,
                                         verbose=verbose)
                   for sfp, nfp in zip(speech_file_path, noise_file_path)]
        if relevant_only:
            inds_del = [i for i in range(len(out)) if out[i] is None]
            out = np.delete(out, inds_del, axis=0)
            if len(out)>0:
                a, b, c = map(lambda x: np.concatenate(x, axis=0), zip(*[_ for _ in out]))
                out = (a, b, c)
            else:
                out = None
    return out


def store_in_hdf(sound_dir, noise_dir=None, val_split=0.2, n_files=30000,
                 block_size=500, ext='mp4'):
    sound_paths, noise_paths = get_file_paths(sound_dir, noise_dir=noise_dir,
                                              n_files=n_files, ext=ext)
    sound_paths_sub = ft.sublists_from_list(sound_paths, block_size)
    noise_paths_sub = ft.sublists_from_list(noise_paths, block_size)
    dir_store = util.apply_recursively(lambda p: os.path.split(p)[0], sound_paths[0])
    dir_store = os.path.join(dir_store, f'storage_{util.timestamp("day")}')
    if not os.path.exists(dir_store):
        os.makedirs(dir_store, exist_ok=True)
    path_hFile = os.path.join(dir_store, f'stored_data_{util.timestamp("day")}.h5')
    nBlocks = len(sound_paths_sub)
    iBlock = 0
    for sound_paths_, noise_paths_ in zip(sound_paths_sub, noise_paths_sub):
        print(f'Block {iBlock+1}/{len(sound_paths_sub)}')
        iBlock += 1
        out = preprocess_video_pair(sound_paths_, noise_paths_, verbose=True)
        inds_del = [i for i in range(len(out)) if out[i] is None]
        if len(inds_del) > 0:
            print(f'Could not read {inds_del} files!')
        mixed_spectrograms, vid_samples, sound_spectrograms = \
            map(lambda x: np.delete(x, inds_del, axis=0), out)
        mixed_spectrograms, vid_samples, sound_spectrograms = \
            map(np.array, (mixed_spectrograms, vid_samples, sound_spectrograms))
        n_samples = mixed_spectrograms.shape[0]
        inds_all = np.arange(n_samples)
        np.random.shuffle(inds_all)
        n_samples_val = int(n_samples * val_split)
        inds_val = inds_all[:n_samples_val]
        inds_train = inds_all[n_samples_val:]
        print(f'{len(inds_val)}/{len(inds_all)} validation samples')
        mixed_spectrograms_train, vid_samples_train, sound_spectrograms_train = \
            mixed_spectrograms[inds_train], vid_samples[inds_train], \
            sound_spectrograms[inds_train]
        mixed_spectrograms_val, vid_samples_val, sound_spectrograms_val = \
            mixed_spectrograms[inds_val], vid_samples[inds_val], \
            sound_spectrograms[inds_val]

        print('Normalizing videos..')
        vid_normalizer = VideoNormalizer(vid_samples_train)
        vid_normalizer.normalize(vid_samples_train)
        vid_normalizer.normalize(vid_samples_val)

        keyNames = ('mixed_spectrograms_train', 'vid_samples_train',
                    'sound_spectrograms_train', 'mixed_spectrograms_val',
                    'vid_samples_val', 'sound_spectrograms_val')
        with h5py.File(path_hFile, mode='a') as hFile:
            if iBlock == 0:
                for key in keyNames:
                    if key in hFile:
                        del hFile[key]
            else:
                for key in keyNames:
                    hFile = ft.createOrAppendToHdf(hFile, key, eval(key), verbose=True)
        # print('Saving normalizer...')
        # path_normalizer = os.path.join(dir_store, 'normalizer.pkl')
        # with open(path_normalizer, mode='wb') as norm_file:
        #     pickle.dump(vid_normalizer, norm_file)
    return path_hFile


class VideoNormalizer(object):
    def __init__(self, video_samples):
        # video_samples: slices x height x width x frames_per_slice
        self.__mean_image = np.mean(video_samples, axis=(0, 3)).astype(np.float32)
        self.__std_image = np.std(video_samples, axis=(0, 3)).astype(np.float32)

    def normalize(self, video_samples):
        video_samples = video_samples.astype(np.float32)
        for slc in range(video_samples.shape[0]):
            for frame in range(video_samples.shape[3]):
                video_samples[slc, :, :, frame] -= self.__mean_image
                video_samples[slc, :, :, frame] /= self.__std_image

    def save(self, path):
        with open(path, mode='wb') as vid_file:
            pickle.dump(self, vid_file)
        print(f'Saved at \n{path}')


