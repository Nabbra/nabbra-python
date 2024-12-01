import scipy.fft
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple


class AudioAmplifier:
    def __init__(
        self,
        sample_rate: int = 8000,
        chunk_size: int = 1024,
        ranges: Dict[Tuple[int, int], int] = {},
        linear_gain: bool = False,
    ):
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._ranges = ranges
        self._linear_gain = linear_gain

    def _apply_gain_to_ranges(self, audio: np.ndarray) -> np.ndarray:
        fft_data, frequencies = self._get_fft_frequencies(audio)

        for (freq_min, freq_max), gain in self._ranges.items():
            fft_data = self._apply_gain_to_range(
                frequencies=frequencies,
                data=fft_data,
                freq_min=freq_min,
                freq_max=freq_max,
                gain=gain
            )

        amplified_audio = self._restore_amplified_data(fft_data)
        return self._float32_to_int16(amplified_audio)

    def _apply_gain_to_range(
        self,
        frequencies: np.ndarray,
        data: np.ndarray,
        freq_min: int,
        freq_max: int,
        gain: int,
    ):
        mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        data[mask] *= self._gain_to_linear(gain) if not self._linear_gain else gain
        return data

    def apply_gain_from_bytes(self, data: bytes) -> np.ndarray:
        audio = self._collect_from_bytes(data)
        return self._apply_gain_to_ranges(audio)

    def apply_gain_from_numpy(self, audio: np.ndarray) -> np.ndarray:
        return self._apply_gain_to_ranges(audio)

    def _collect_from_bytes(self, data: bytes) -> np.ndarray:
        return self._int16_to_float32(np.frombuffer(data, dtype=np.int16))

    def _get_fft_frequencies(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fft_data = scipy.fft.fft(audio)
        frequencies = scipy.fft.fftfreq(len(audio), 1 / self._sample_rate)
        return fft_data, frequencies

    def _restore_amplified_data(self, data: np.ndarray) -> np.ndarray:
        return scipy.fft.ifft(data).real

    def _gain_to_linear(self, gain: float) -> float:
        return 10 ** (gain / 20.0)

    def set_sample_rate(self, rate: int) -> None:
        self._sample_rate = rate

    def set_chunk_size(self, size: int) -> None:
        self._chunk_size = size

    def _int16_to_float32(self, audio_data):
        return audio_data.astype(np.float32) / 32768.0

    def _float32_to_int16(self, audio_data):
        audio_data = np.clip(audio_data * 32768.0, -32768, 32767)
        return audio_data.astype(np.int16)

    def set_ranges(self, ranges: Dict[Tuple[int, int], int] = {}) -> None:
        self._ranges = ranges

    def to_linear_gain(self, value: bool = True):
        self._linear_gain = value

    def plot_frequency_spectrum(self, original_audio: np.ndarray, amplified_audio: np.ndarray, filename: str):
        original_fft, original_frequencies = self._get_fft_frequencies(original_audio)
        amplified_fft, amplified_frequencies = self._get_fft_frequencies(amplified_audio)

        original_magnitude = np.abs(original_fft)
        amplified_magnitude = np.abs(amplified_fft)

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(original_frequencies[:len(original_frequencies)//2], original_magnitude[:len(original_magnitude)//2])
        plt.title('Original Audio Frequency Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')

        plt.xlim(250, 1000)
        plt.ylim(0, 30000)

        plt.subplot(2, 1, 2)
        plt.plot(amplified_frequencies[:len(amplified_frequencies)//2], amplified_magnitude[:len(amplified_magnitude)//2])
        plt.title('Amplified Audio Frequency Spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')

        plt.xlim(250, 1000)
        plt.ylim(0, 30000)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()