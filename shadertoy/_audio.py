import numpy as np
import io
import os
import threading
import sounddevice as sd
import soundfile as sf
import requests
from tqdm import tqdm

class NumpyCircularBuffer:
    """
    Circular buffer implemented using numpy arrays.
    This is used to store the last N samples of audio data.
    """

    def __init__(self, max_size, data_shape, dtype=np.float32):
        self.max_size = max_size
        self.data_shape = data_shape
        self.buffer = np.zeros((max_size,) + data_shape, dtype=dtype)
        self.index = 0

    def append(self, data):
        num_items = data.shape[0]
        if num_items > self.max_size:
            raise ValueError("The input data is larger than the buffer size.")

        # Calculate the insertion index range
        end_index = (self.index + num_items) % self.max_size

        if end_index > self.index:
            self.buffer[self.index : end_index] = data
        else:
            # Wrap-around case
            part1_size = self.max_size - self.index
            self.buffer[self.index :] = data[:part1_size]
            self.buffer[:end_index] = data[part1_size:]

        self.index = end_index

    def get_last_n(self, n):
        start_index = (self.index - n) % self.max_size
        if start_index < 0:
            start_index += self.max_size

        if start_index < self.index:
            return self.buffer[start_index : self.index]
        else:
            return np.concatenate(
                (self.buffer[start_index:], self.buffer[: self.index]), axis=0
            )


class AudioAnalyzer:
    """
    Simple implementation of Audio AnalyserNode in W3C Web Audio API.
    See: https://www.w3.org/TR/webaudio/#AnalyserNode
    """

    def __init__(
        self, fft_size=1024, min_decibels=-100, max_decibels=-30, smoothing_factor=0.8
    ):
        self.fft_size = fft_size
        self.min_decibels = min_decibels
        self.max_decibels = max_decibels
        self.smoothing_factor = smoothing_factor

        # last 32768 samples, 2 channels
        # In order to allow for an increase in fftsize, we should effectively keep around the last 32768 samples
        self._buffer = NumpyCircularBuffer(32768, (2,))

    @property
    def frequency_bin_count(self):
        return self.fft_size // 2

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, value):
        assert value <= 32768 and value >= 32, "fft_size must be between 32 and 32768"
        assert value & (value - 1) == 0, "fft_size must be a power of 2"
        self._fft_size = value
        self._frequency_data = np.zeros(self.fft_size // 2 + 1, dtype=np.float32)
        self._byte_frequency_data = np.zeros(self.fft_size // 2 + 1, dtype=np.uint8)
        self.__blackman_window = self._get_blackman_window()
        self.__previous_smoothed_data = np.zeros(value // 2 + 1, dtype=np.float32)

    def _get_blackman_window(self):
        a0 = 0.42
        a1 = 0.5
        a2 = 0.08
        n = np.arange(self.fft_size, dtype=np.float32)
        w = (
            a0
            - a1 * np.cos(2 * np.pi * n / self.fft_size)
            + a2 * np.cos(4 * np.pi * n / self.fft_size)
        )  # W3C spec use N not N-1, so we use fft_size not fft_size-1 here, todo: confirm is it correct?
        return w

    def receive_data(self, data):
        self._buffer.append(data)

        # A block of 128 samples-frames is called a render quantum
        # within the same render quantum as a previous call, the current frequency data is not updated with the same data.
        # Instead, the previously computed data is returned.
        # we assume that len(data) always >= 128
        self._byte_frequency_data = None
        self._frequency_data = None

    def get_time_domain_data(self):
        time_domain_data = self._buffer.get_last_n(self.fft_size)
        time_domain_data = np.mean(time_domain_data, axis=1, dtype=np.float32)
        # the data should be already in range -1 to 1, but we clip it just in case of any overflow
        time_domain_data = np.clip(time_domain_data, -1, 1)
        return time_domain_data

    def get_byte_time_domain_data(self):
        time_domain_data = self.get_time_domain_data()
        return np.floor((time_domain_data + 1) * 128).astype(np.uint8)

    def get_frequency_data(self):
        if self._frequency_data is None:

            time_domain_data = self.get_time_domain_data()

            frames_windowed = time_domain_data * self.__blackman_window
            # Perform FFT
            # def _fft(data):
            #     N = len(data)
            #     W = np.exp(-2j * np.pi / N)
            #     X = np.zeros(N // 2 + 1, dtype=complex)
            #     for k in range(N // 2 + 1):
            #         for n in range(N):
            #             X[k] += data[n] * W**(k * n)
            #         X[k] /= N
            #     return X
            fft_result = np.fft.rfft(frames_windowed, n=self.fft_size) / self.fft_size

            # Smooth over time
            smoothing_factor = self.smoothing_factor
            smoothed_data = smoothing_factor * self.__previous_smoothed_data + (
                1 - smoothing_factor
            ) * np.abs(fft_result)

            # Handle non-finite values
            smoothed_data = np.nan_to_num(
                smoothed_data, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Update previous smoothed data
            self.__previous_smoothed_data = smoothed_data

            # Convert to dB
            self._frequency_data = 20 * np.log10(smoothed_data)

        return self._frequency_data

    def get_byte_frequency_data(self):
        if self._byte_frequency_data is None:

            frequency_data = self.get_frequency_data()

            clipped_data = np.clip(frequency_data, self.min_decibels, self.max_decibels)
            scale = 255 / (self.max_decibels - self.min_decibels)
            self._byte_frequency_data = np.floor(
                (clipped_data - self.min_decibels) * scale
            ).astype(np.uint8)

        return self._byte_frequency_data


class AudioPlayer:
    def __init__(self) -> None:
        self._analyzer = None
        # we use 128 samples as a block size as default, it's called a render quantum in W3C spec
        self._block_size = 128
        self._cache_block_size = 50 * 1024
        self._mini_playable_size = 10 * 1024

    @property
    def block_size(self):
        return self._block_size

    @property
    def analyzer(self):
        return self._analyzer

    @analyzer.setter
    def analyzer(self, analyzer):
        self._analyzer = analyzer

    def play(self, path, stream=True):
        if "https://" in str(path) or "http://" in str(path):
            if stream:
                play_func = self._play_stream
            else:
                r = requests.get(path)
                r.raise_for_status()
                path = io.BytesIO(r.content)
                play_func = self._play_local
        else:
            play_func = self._play_local

        play_t = threading.Thread(target=play_func, args=(path,), daemon=True)
        play_t.start()

    def _play_local(self, local_file):
        data, samplerate = sf.read(local_file, dtype=np.float32, always_2d=True)
        block_size = self.block_size

        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=data.shape[1],
            dtype=np.float32,
            blocksize=block_size,
        )

        with stream:
            length = len(data)
            with tqdm(
                total=length / samplerate + 0.001,
                unit="s",
                unit_scale=True,
                desc="Playing",
            ) as pbar:
                for i in range(0, length, block_size):
                    frames_data = data[i : min(i + block_size, length)]
                    stream.write(frames_data)
                    if self.analyzer:
                        self.analyzer.receive_data(frames_data)
                    pbar.update(block_size / samplerate)

    def _play_stream(self, path):
        response = requests.get(path, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        audio_data = io.BytesIO()
        bytes_lock = threading.Lock()
        block_data_available = threading.Event()

        def _download_data():
            chunk_size = 1024
            playback_block_size = self._cache_block_size
            mini_playable_size = self._mini_playable_size
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as dbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    # time.sleep(0.05) # simulate slow download
                    with bytes_lock:
                        last_read_pos = audio_data.tell()
                        audio_data.seek(0, os.SEEK_END)
                        audio_data.write(chunk)
                        end_pos = audio_data.tell()
                        audio_data.seek(0)
                        audio_data.seek(last_read_pos)
                    if end_pos - last_read_pos > playback_block_size:
                        block_data_available.set()  # resume playback if buffer is enough
                    elif end_pos - last_read_pos <= mini_playable_size:
                        block_data_available.clear()  # pause playback if not enough data
                    dbar.update(len(chunk))
                block_data_available.set()

        download_data_t = threading.Thread(target=_download_data, daemon=True)
        download_data_t.start()

        # wait for the first block of data to be available, then create the soundFile
        while True:
            try:
                block_data_available.wait()
                with bytes_lock:
                    audio_data.seek(0)
                    audio_file = sf.SoundFile(audio_data, mode="r")
                    break
            except Exception:
                block_data_available.clear()

        block_size = self.block_size

        stream = sd.OutputStream(
            samplerate=audio_file.samplerate,
            channels=audio_file.channels,
            dtype=np.float32,
            blocksize=block_size,
        )

        total_time = audio_file.frames / audio_file.samplerate + 0.001
        with stream:
            with tqdm(
                total=total_time, unit="s", unit_scale=True, desc="Playing"
            ) as pbar:
                while True:
                    block_data_available.wait()
                    with bytes_lock:
                        data = audio_file.read(
                            block_size, dtype=np.float32, always_2d=True
                        )
                    if len(data) > 0:
                        stream.write(data)
                        if self.analyzer:
                            self.analyzer.receive_data(data)
                        pbar.update(len(data) / audio_file.samplerate)
                    else:
                        break
