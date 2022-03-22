"""
Classes for better code structure.
References:
    [1] Stein et al., Automatic Detection of Audio Effects in Guitar and Bass Recordings, AES 2010.
"""
import pathlib
import warnings

import numpy as np
import sounddevice
import soundfile
from numpy.typing import ArrayLike
from playsound import playsound

import features
import util


class SoundSample:
    """
    A class to represent a sound sample with all its features and relevant information.
    """

    @staticmethod
    def _idmt_instrument(char: str):
        match char:
            case 'B':
                return 'bass'
            case 'G':
                return 'guitar'
            case _:
                return None

    @staticmethod
    def _idmt_model(char: str):
        match char:
            case '1':
                return 'Yamaha BB604 - 1'
            case '2':
                return 'Yamaha BB604 - 2'
            case '3':
                return 'Warwick Corvette - 1'
            case '4':
                return 'Warwick Corvette - 2'
            case '5':
                return 'Undefined'
            case '6':
                return 'Schecter Diamond C-1 Classic - 1'
            case '7':
                return 'Schecter Diamond C-1 Classic - 2'
            case '8':
                return 'Chester Stratocaster - 1'
            case '9':
                return 'Chester Stratocaster - 2'
            case _:
                return None

    @staticmethod
    def _idmt_playing(char: str):
        match char:
            case '1':
                return 'Finger light'
            case '2':
                return 'Finger hard'
            case '3':
                return 'Plectrum'
            case _:
                return None

    @staticmethod
    def _idmt_pitch_midi(char: str):
        return int(char)

    @staticmethod
    def _idmt_string(char: str):
        match char:
            case '1':
                return 'E'
            case '2':
                return 'A'
            case '3':
                return 'D'
            case '4':
                return 'G'
            case '5':
                return 'B'
            case '6':
                return 'e'
            case _:
                return None

    @staticmethod
    def _idmt_fret(char: str):
        return int(char)

    @staticmethod
    def _idmt_fx_type(char: str):
        match char:
            case '1':
                return 'No Fx'
            case '2':
                return 'Ambient'
            case '3':
                return 'Modulation'
            case '4':
                return 'Distortion'
            case _:
                return None

    @staticmethod
    def _idmt_fx(char: str):
        match char:
            case '11':
                return 'Dry'
            case '12':
                return 'Amp sim'
            case '21':
                return 'Feedback delay'
            case '22':
                return 'Slapback delay'
            case '23':
                return 'Reverb'
            case '31':
                return 'Chorus'
            case '32':
                return 'Flanger'
            case '33':
                return 'Phaser'
            case '34':
                return 'Tremolo'
            case '35':
                return 'Vibrato'
            case '41':
                return 'Distortion'
            case '42':
                return 'Overdrive'
            case _:
                return None

    @staticmethod
    def _idmt_setting(char: str):
        return int(char)

    @staticmethod
    def _idmt_id(char: str):
        return int(char)

    @staticmethod
    def idmt_parsing(name: str) -> dict:
        """
        Returns a dictionary with the information contained in the input string according to [1] specification.
        """
        dic = {}
        info = name.split('-')
        dic['instrument'] = SoundSample._idmt_instrument(info[0][0])
        dic['model'] = SoundSample._idmt_model(info[0][1])
        dic['playing'] = SoundSample._idmt_playing(info[0][2])
        dic['midi_pitch'] = SoundSample._idmt_pitch_midi(info[1][:1])
        dic['string'] = SoundSample._idmt_string(info[1][2])
        dic['fret'] = SoundSample._idmt_fret(info[1][3:])
        dic['fx_type'] = SoundSample._idmt_fx_type(info[2][0])
        dic['fx'] = SoundSample._idmt_fx(info[2][1:3])
        dic['setting'] = SoundSample._idmt_setting(info[2][3])
        dic['id'] = SoundSample._idmt_id(info[3])
        return dic

    def __init__(self, data: str or pathlib.Path or ArrayLike = None, filename: str or pathlib.Path = None,
                 rate: float = None, idmt: bool = False) -> None:
        """
        :param data: string, pathlib.Path or float array representing either where to find the audio file or directly
                     the audio samples. If array, rate must be set.
        :param filename: Path to audio file.
        :param rate: Sampling rate of the audio signal. Mandatory if data is already the audio waveform.
        :param idmt: Is the file from the IDMT-SMT dataset? Default is False.
        """
        if data is None and filename is None:
            raise ValueError("Cannot instantiate an empty SoundSample. Audio data must be given.")
        if data is not None and isinstance(data, (np.ndarray, list)):
            if rate is None:
                raise ValueError("Rate must be specified if data is an array of audio samples.")
            self.data = data
            self.rate = rate
            if filename is not None:
                self.file = pathlib.Path(filename)
            else:
                self.file = None
        if data is not None and isinstance(data, (str, pathlib.Path)):
            self.file = pathlib.Path(data)
            self.data, self.rate = util.read_audio(str(self.file), add_noise=idmt)
        if data is None:
            self.file = pathlib.Path(filename)
            self.data, self.rate = soundfile.read(self.file)
        self.idmt_info = None
        if idmt:
            if self.file is None:
                raise UserWarning("IDMT dataset information cannot be retrieved if no filename is given.")
            else:
                self.set_idmt_info()
        self.spectral_features = {}
        self.stft = None
        self.stft_freq, self.stft_times = None, None
        self.mag, self.phase = None, None
        return

    def set_idmt_info(self) -> None:
        """
        Extract information from IDMT-SMT filename
        """
        name = self.file.name.split('.')[0]
        self.idmt_info = SoundSample.idmt_parsing(name)
        return

    def write(self, path: str or pathlib.Path = './sound.wav', overwrite: bool = False) -> None:
        path = pathlib.Path(path)
        if path.is_dir():
            warnings.warn("Path is a directory. File is written under the default name 'sound.wav'.", UserWarning)
            path = path / 'sound.wav'
        if path.suffix == '':
            path = path.with_suffix('.wav')
        if path.exists() and not overwrite:
            raise ValueError("File already exists, change path or set overwrite to True.")
        soundfile.write(path, self.data, self.rate)
        return

    def play(self) -> None:
        if self.file is not None:
            playsound(self.file, block=True)
        else:
            sounddevice.play(data=self.data, samplerate=self.rate, blocking=True)
        return

    def set_stft(self, fft_size: int, **kwargs) -> None:
        self.stft, self.stft_freq, self.stft_times = util.get_stft(self.data, self.rate, fft_size, **kwargs)
        self.mag = np.abs(self.stft)
        self.phase = np.angle(self.stft)
        return

    def set_spectral_features(self, flux_q_norm: int = 1, rolloff_thresh: float = 0.95,
                              flat_bands: int or list = 1) -> None:
        if self.stft is None:
            raise ValueError('Call set_stft before setting spectral features.')
        self.spectral_features['centroid'] = features.spectral_centroid(mag=self.mag, freq=self.stft_freq)
        self.spectral_features['spread'] = features.spectral_spread(mag=self.mag, freq=self.stft_freq,
                                                                    cent=self.spectral_features['centroid'])
        self.spectral_features['skewness'] = features.spectral_skewness(mag=self.mag, freq=self.stft_freq,
                                                                        cent=self.spectral_features['centroid'])
        self.spectral_features['kurtosis'] = features.spectral_kurtosis(mag=self.mag, freq=self.stft_freq,
                                                                        cent=self.spectral_features['centroid'])
        self.spectral_features['flux'] = features.spectral_flux(mag=self.mag, q_norm=flux_q_norm)
        self.spectral_features['rolloff'] = features.spectral_rolloff(mag=self.mag, freq=self.stft_freq,
                                                                      threshold=rolloff_thresh)
        self.spectral_features['slope'] = features.spectral_slope(mag=self.mag, freq=self.stft_freq)
        self.spectral_features['flatness'] = features.spectral_flatness(mag=self.mag, rate=self.rate, bands=flat_bands)
        return

    @property
    def spectral_centroid(self) -> ArrayLike:
        if 'centroid' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            self.spectral_features['centroid'] = features.spectral_centroid(mag=self.mag, freq=self.stft_freq)
        return self.spectral_features['centroid']

    @property
    def spectral_spread(self) -> ArrayLike:
        if 'spread' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            if 'centroid' not in self.spectral_features:
                self.spectral_features['spread'] = features.spectral_spread(mag=self.mag, freq=self.stft_freq)
            else:
                self.spectral_features['spread'] = features.spectral_spread(mag=self.mag, freq=self.stft_freq,
                                                                            cent=self.spectral_features['centroid'])
        return self.spectral_features['spread']

    @property
    def spectral_skewness(self) -> ArrayLike:
        if 'skewness' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            if 'centroid' not in self.spectral_features:
                self.spectral_features['skewness'] = features.spectral_skewness(mag=self.mag, freq=self.stft_freq)
            else:
                self.spectral_features['skewness'] = features.spectral_skewness(mag=self.mag, freq=self.stft_freq,
                                                                                cent=self.spectral_features['centroid'])
        return self.spectral_features['skewness']

    @property
    def spectral_kurtosis(self) -> ArrayLike:
        if 'kurtosis' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            if 'centroid' not in self.spectral_features:
                self.spectral_features['kurtosis'] = features.spectral_kurtosis(mag=self.mag, freq=self.stft_freq)
            else:
                self.spectral_features['kurtosis'] = features.spectral_kurtosis(mag=self.mag, freq=self.stft_freq,
                                                                                cent=self.spectral_features['centroid'])
        return self.spectral_features['kurtosis']

    @property
    def spectral_flux(self) -> ArrayLike:
        if 'flux' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            warnings.warn("Spectral flux computed with default values.", UserWarning)
            self.spectral_features['flux'] = features.spectral_flux(mag=self.mag)
        return self.spectral_features['flux']

    @property
    def spectral_rolloff(self) -> ArrayLike:
        if 'rolloff' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            warnings.warn("Spectral roll-off computed with default values.", UserWarning)
            self.spectral_features['rolloff'] = features.spectral_rolloff(mag=self.mag, freq=self.stft_freq)
        return self.spectral_features['rolloff']

    @property
    def spectral_slope(self) -> ArrayLike:
        if 'slope' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            self.spectral_features['slope'] = features.spectral_slope(mag=self.mag, freq=self.stft_freq)
        return self.spectral_features['slope']

    @property
    def spectral_flatness(self) -> ArrayLike:
        if 'flatness' not in self.spectral_features:
            if self.stft is None:
                raise ValueError('Call set_stft before accessing spectral features.')
            warnings.warn("Spectral flatness computed with default values.", UserWarning)
            self.spectral_features['flatness'] = features.spectral_flatness(mag=self.mag, rate=self.rate)
        return self.spectral_features['flatness']

    @property
    def instrument(self) -> str:
        return self.idmt_info['instrument']

    @property
    def model(self) -> str:
        return self.idmt_info['model']

    @property
    def playing(self) -> str:
        return self.idmt_info['playing']

    @property
    def midi_pitch(self) -> int:
        return self.idmt_info['midi_pitch']

    @property
    def string(self) -> str:
        return self.idmt_info['string']

    @property
    def fret(self) -> int:
        return self.idmt_info['fret']

    @property
    def fx_type(self) -> str:
        return self.idmt_info['fx_type']

    @property
    def fx(self) -> str:
        return self.idmt_info['fx']

    @property
    def setting(self) -> int:
        return self.idmt_info['setting']

    @property
    def id(self) -> int:
        return self.idmt_info['id']

    @property
    def pitch(self) -> float:
        """
        Fundamental pitch in Hertz.
        """
        return 440 * 2 ** ((self.midi_pitch - 69) / 12)
