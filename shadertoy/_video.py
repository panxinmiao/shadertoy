import wgpu
from ._channel import ShadertoyChannel
from ._shared import get_device
import time

class VideoChannel(ShadertoyChannel):
    def __init__(self, uri, filter="linear", wrap="clamp", vflip=False) -> None:
        self._uri = uri

        from moviepy import VideoFileClip
        self._clip = VideoFileClip(uri, audio=False, has_mask=True)

        self._device = get_device()
        texture = self._device.create_texture(
            size=self._clip.size,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._time = 0.0
        self._start_time = 0.0
        self._vlip = vflip
        self._is_playing = False
        super().__init__(texture, filter, wrap)

    def play(self):
        self._last_play_time = time.perf_counter()
        self._is_playing = True

    @property
    def time(self):
        return self._time
    
    def _set_play_time(self, t):
        self._time = t

    def update(self):
        if self._is_playing:
            now = time.perf_counter()
            self._time += now - self._last_play_time
            self._last_play_time = now

        import numpy as np
        self._clip.has_mask = True

        data = self._clip.reader.get_frame(self.time)

        if self._vlip:
            data = np.flipud(data)

        data = np.ascontiguousarray(data)

        size = self._clip.size
        bytes_per_row = 4 * size[0]

        self._device.queue.write_texture(
            {"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": bytes_per_row},
            size,
        )