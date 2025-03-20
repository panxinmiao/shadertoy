import time
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
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
        resolution=(800, 450),
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
        self._uniform_data["resolution"] = resolution + (1,)

        self._canvas = WgpuCanvas(title=title, size=resolution, max_fps=60)
        self._device = get_device()
        self._canvas_context = self._canvas.get_context()

        # We use "bgra8unorm" not "bgra8unorm-srgb" here because we want to let the shader fully control the color-space.
        self._canvas_context.configure(
            device=self._device, format=wgpu.TextureFormat.bgra8unorm
        )
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
            buffer_a = BufferChannel(resolution)
            self._buffer_a_pass = ShaderPass(
                common_code + buffer_a_code, render_target=buffer_a
            )
        if buffer_b_code:
            buffer_b = BufferChannel(resolution)
            self._buffer_b_pass = ShaderPass(
                common_code + buffer_b_code, render_target=buffer_b
            )
        if buffer_c_code:
            buffer_c = BufferChannel(resolution)
            self._buffer_c_pass = ShaderPass(
                common_code + buffer_c_code, render_target=buffer_c
            )
        if buffer_d_code:
            buffer_d = BufferChannel(resolution)
            self._buffer_d_pass = ShaderPass(
                common_code + buffer_d_code, render_target=buffer_d
            )

        if sound_code:
            self._sound_pass = SoundPass(common_code + sound_code)

        self._main_pass = ShaderPass(
            common_code + main_code, render_target=self._canvas, flip=True
        )

        self._bind_events()

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

    def _draw_frame(self):
        # Update uniform buffer
        self._update()

        command_encoder = self._device.create_command_encoder()

        passes = [
            self._buffer_a_pass,
            self._buffer_b_pass,
            self._buffer_c_pass,
            self._buffer_d_pass,
            self._main_pass,
        ]

        for pass_ in passes:
            if pass_ is not None:
                pass_.draw_frame(command_encoder, self._uniform_buffer_bind_group)

        self._device.queue.submit([command_encoder.finish()])
        self._canvas.request_draw()

    def _bind_events(self):
        def on_resize(event):
            w, h = int(event["width"]), int(event["height"])
            if w == 0 or h == 0:
                return
            self._uniform_data["resolution"] = (w, h, 1)
            passes = [
                self._buffer_a_pass,
                self._buffer_b_pass,
                self._buffer_c_pass,
                self._buffer_d_pass,
            ]
            for pass_ in passes:
                if pass_ is not None:
                    pass_.render_target.resize((w, h))

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

    def _update(self):
        now = time.perf_counter()
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

    def show(self):
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

        self._canvas.request_draw(self._draw_frame)
        run()
