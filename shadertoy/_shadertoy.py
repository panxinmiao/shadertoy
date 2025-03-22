import time
import numpy as np
import wgpu
from importlib.util import find_spec
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
from ._shared import get_device, get_uniform_input_layout
from ._channel import BufferChannel
from ._pass import ShaderPass
from ._sound_pass import SoundPass

class Shadertoy:
    def __init__(
        self,
        main_code,
        common_code=None,
        buffer_a_code=None,
        buffer_b_code=None,
        buffer_c_code=None,
        buffer_d_code=None,
        sound_code=None,
        title="Shadertoy",
    ) -> None:

        self._uniform_data = np.zeros(
            (),
            dtype=[
                ("mouse", "float32", (4)),
                ("time", "float32"),
                ("time_delta", "float32"),
                ("frame", "uint32"),
                ("frame_rate", "float32"),
                ("date", "float32", (4)),
                ("resolution", "float32", (3)),
                ("__padding", "uint32"),  # Padding to 64
            ],
        )

        self._title = title
        self._device = get_device()

        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._uniform_buffer_bind_group = self._device.create_bind_group(
            layout=get_uniform_input_layout(),
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniform_buffer,
                        "offset": 0,
                        "size": self._uniform_data.nbytes,
                    },
                },
            ],
        )

        self._buffer_a_pass = None
        self._buffer_b_pass = None
        self._buffer_c_pass = None
        self._buffer_d_pass = None
        self._sound_pass = None

        common_code = common_code or ""

        if buffer_a_code:
            self._buffer_a_pass = ShaderPass(
                f"{common_code}\n{buffer_a_code}", render_target=BufferChannel()
            )
        if buffer_b_code:
            self._buffer_b_pass = ShaderPass(
                f"{common_code}\n{buffer_b_code}", render_target=BufferChannel()
            )
        if buffer_c_code:
            self._buffer_c_pass = ShaderPass(
                f"{common_code}\n{buffer_c_code}", render_target=BufferChannel()
            )
        if buffer_d_code:
            self._buffer_d_pass = ShaderPass(
                f"{common_code}\n{buffer_d_code}", render_target=BufferChannel()
            )

        if sound_code:
            self._sound_pass = SoundPass(f"{common_code}\n{sound_code}")

        self._main_pass = ShaderPass(
            f"{common_code}\n{main_code}", flip=True
        )

        # default resolution
        self._uniform_data["resolution"] = (800, 450, 1)


    @property
    def resolution(self):
        return tuple(self._uniform_data["resolution"][:2])

    @property
    def main_pass(self):
        return self._main_pass

    @property
    def buffer_a_pass(self):
        return self._buffer_a_pass

    @property
    def buffer_b_pass(self):
        return self._buffer_b_pass

    @property
    def buffer_c_pass(self):
        return self._buffer_c_pass

    @property
    def buffer_d_pass(self):
        return self._buffer_d_pass
    
    @property
    def sound_pass(self):
        return self._sound_pass
    
    @property
    def render_target(self):
        return self.main_pass.render_target
    
    @render_target.setter
    def render_target(self, value):
        self.main_pass.render_target = value

    def _draw_frame(self):
        command_encoder = self._device.create_command_encoder()

        buffer_passes = [
            self._buffer_a_pass,
            self._buffer_b_pass,
            self._buffer_c_pass,
            self._buffer_d_pass,
        ]

        for pass_ in buffer_passes:
            if pass_ is not None:
                pass_.render_target.ensure_texture(self.resolution)

        for pass_ in [*buffer_passes, self._main_pass]:
            if pass_ is not None:
                pass_.draw_frame(command_encoder, self._uniform_buffer_bind_group)

        self._device.queue.submit([command_encoder.finish()])


    def _bind_events(self):
        def on_resize(event):
            w, h = int(event["width"]), int(event["height"])
            if w == 0 or h == 0:
                return
            self._uniform_data["resolution"] = (w, h, 1)

        def on_mouse_move(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                xy = event["x"], self.resolution[1] - event["y"]
                self._uniform_data["mouse"][:2] = xy

        def on_mouse_down(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                x, y = event["x"], self.resolution[1] - event["y"]
                self._uniform_data["mouse"] = (x, y, x, y)

        def on_mouse_up(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                self._uniform_data["mouse"][2] = -abs(self._uniform_data["mouse"][2])

        self._canvas.add_event_handler(on_resize, "resize")
        self._canvas.add_event_handler(on_mouse_move, "pointer_move")
        self._canvas.add_event_handler(on_mouse_down, "pointer_down")
        self._canvas.add_event_handler(on_mouse_up, "pointer_up")

    def _update(self, t=None):
        if t is None:
            now = time.perf_counter()
        else:
            now = t

        if not hasattr(self, "_last_time"):
            self._last_time = now
        time_delta = now - self._last_time
        self._uniform_data["time_delta"] = time_delta
        self._last_time = now
        self._uniform_data["time"] += time_delta

        if not hasattr(self, "_frame"):
            self._frame = 0

        self._uniform_data["frame"] = self._frame
        self._frame += 1

        # fps
        if not hasattr(self, "_fps"):
            self._fps = now, 1

        if now > self._fps[0] + 1:
            fps = self._fps[1] / (now - self._fps[0])
            self._uniform_data["frame_rate"] = fps
            self._fps = now, 1
        else:
            self._fps = self._fps[0], self._fps[1] + 1

        current_time = time.time()
        time_struct = time.localtime(current_time)

        self._uniform_data["date"] = (
            float(time_struct.tm_year - 1),
            float(time_struct.tm_mon - 1),
            float(time_struct.tm_mday),
            time_struct.tm_hour * 3600
            + time_struct.tm_min * 60
            + time_struct.tm_sec
            + current_time % 1,
        )

        if self._uniform_data["mouse"][3] > 0:
            if getattr(self, "_clicked", False):
                self._clicked = False
                self._uniform_data["mouse"][3] = -abs(self._uniform_data["mouse"][3])
            else:
                self._clicked = True

        self._device.queue.write_buffer(
            self._uniform_buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

    def set_shader_state(
        self, 
        time: float = 0.0,
        time_delta: float = 0.167,
        frame: int = 0,
        framerate: int = 60.0,
        mouse_pos: tuple = None,
        date: tuple = None,
    ):
        self._uniform_data["time"] = time
        self._uniform_data["time_delta"] = time_delta
        self._uniform_data["frame"] = frame
        self._uniform_data["frame_rate"] = framerate
        if mouse_pos is not None:
            self._uniform_data["mouse"] = mouse_pos
        if date is not None:
            self._uniform_data["date"] = date

        self._device.queue.write_buffer(
            self._uniform_buffer, 0, self._uniform_data, 0, self._uniform_data.nbytes
        )

    def snapshot(self):
        # if don't have a render target, it probably means we are running in headless mode
        # so we create an offscreen canvas to render the frame
        if self.render_target is None:
            self._canvas = OffscreenCanvas(size = self.resolution)
            self._canvas_context = self._canvas.get_context()
            self._canvas_context.configure(
                device = self._device, format = wgpu.TextureFormat.rgba8unorm, usage = wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT
            )
            self.render_target = self._canvas_context

        self._draw_frame()
        texture = self.main_pass.render_target.get_current_texture()
        size = texture.size
        bytes_per_pixel = 4

        data = self._device.queue.read_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )
        return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)
    
    def to_video(self, out, duration: int = 10, fps: int = 60, resolution = (1280, 720), **kwargs):
        if not find_spec("moviepy"):
            raise ImportError("Please install moviepy to use this feature.: pip install moviepy")
 
        from ._audio import AudioChannel
        from moviepy import VideoClip, AudioArrayClip, CompositeAudioClip


        if resolution is not None:
            self._uniform_data["resolution"] = resolution + (1,)

        passes = [
            self._sound_pass,
            self._buffer_a_pass,
            self._buffer_b_pass,
            self._buffer_c_pass,
            self._buffer_d_pass,
            self._main_pass,
        ]

        playable_channels = []

        for pass_ in passes:
            if pass_ is not None:
                channels = [pass_.channel_0, pass_.channel_1, pass_.channel_2, pass_.channel_3]
                for channel in channels:
                    if channel and hasattr(channel, "play"):
                        playable_channels.append(channel)


        def frame_function(t):
            if playable_channels:
                for channel in playable_channels:
                    channel._set_play_time(t)

            # self.set_shader_state(time=t)
            self._update(t)
            frame = self.snapshot()
            return frame[:, :, :3]
        

        video_clip = VideoClip(frame_function, duration=duration)

        audio_clips = []

        for channel in playable_channels:
            if isinstance(channel, AudioChannel):
                audio_data, audio_fps = channel._audio._load_data(force_cacahe_stream=True)
                audio_clip = AudioArrayClip(audio_data, fps=audio_fps)
                audio_clip.duration = duration
                audio_clips.append(audio_clip)


        if self.sound_pass:
            audio_data = self.sound_pass.get_audio_data()

            audio_clip = AudioArrayClip(audio_data, fps=44100)
            audio_clip.duration = duration
            audio_clips.append(audio_clip)
        
        if audio_clips:
            audio_clip = CompositeAudioClip(audio_clips)
            audio_clip.duration = duration
            video_clip.audio = audio_clip

        video_clip.write_videofile(out, fps=fps, codec='libx264', **kwargs)

    def show(self, resolution=(800, 450)):
        self._uniform_data["resolution"] = resolution + (1,)
        self._canvas = WgpuCanvas(title=self._title, size=resolution, max_fps=60)
        self._canvas_context = self._canvas.get_context()
        # We use "bgra8unorm" not "bgra8unorm-srgb" here because we want to let the shader fully control the color-space.
        self._canvas_context.configure(
            device=self._device, format=wgpu.TextureFormat.rgba8unorm, usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT
        )
        self.render_target = self._canvas_context
        self._bind_events()

        passes = [
            self._sound_pass,
            self._buffer_a_pass,
            self._buffer_b_pass,
            self._buffer_c_pass,
            self._buffer_d_pass,
            self._main_pass,
        ]

        for pass_ in passes:
            if pass_ is not None:
                pass_.play()

        def loop():
            self._update()
            self._draw_frame()
            self._canvas.request_draw()

        self._canvas.request_draw(loop)
        run()
