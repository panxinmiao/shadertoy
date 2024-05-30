import time
import weakref
import numpy as np
import wgpu
from wgpu.gui.auto import WgpuCanvas, run

vertex_code_glsl = """
#version 450 core

layout(location = 0) out vec2 _uv;

void main(void){
    int index = int(gl_VertexID);
    if (index == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
        _uv = vec2(0.0, 1.0);
    } else if (index == 1) {
        gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
        _uv = vec2(2.0, 1.0);
    } else {
        gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
        _uv = vec2(0.0, -1.0);
    }
}
"""

builtin_variables_glsl = """
#version 450 core

vec3 i_resolution;
vec4 i_mouse;
float i_time;
float i_time_delta;
int i_frame;

// Shadertoy compatibility, see we can use the same code copied from shadertoy website

#define iChannel0 sampler2D(i_channel0, sampler0)
#define iChannel1 sampler2D(i_channel1, sampler1)
#define iChannel2 sampler2D(i_channel2, sampler2)
#define iChannel3 sampler2D(i_channel3, sampler3)

#define iTime i_time
#define iResolution i_resolution
#define iTimeDelta i_time_delta
#define iMouse i_mouse
#define iFrame i_frame

#define mainImage shader_main


layout(set = 1, binding = 0) uniform texture2D i_channel0;
layout(set = 1, binding = 1) uniform sampler sampler0;

layout(set = 2, binding = 0) uniform texture2D i_channel1;
layout(set = 2, binding = 1) uniform sampler sampler1;

layout(set = 3, binding = 0) uniform texture2D i_channel2;
layout(set = 3, binding = 1) uniform sampler sampler2;

layout(set = 4, binding = 0) uniform texture2D i_channel3;
layout(set = 4, binding = 1) uniform sampler sampler3;

layout(location = 0) in vec2 _uv;
struct ShadertoyInput {
    vec4 _mouse;
    vec3 _resolution;
    float _time;
    float _time_delta;
    int _frame;
};

layout(set = 0, binding = 0) uniform ShadertoyInput input;

out vec4 FragColor;
"""

fragment_code_glsl = """

void main(){

    i_time = input._time;
    i_resolution = input._resolution;
    i_time_delta = input._time_delta;
    i_mouse = input._mouse;
    i_frame = input._frame;

    vec2 uv = _uv;
    uv.y = 1.0 - uv.y;
    vec2 frag_coord = uv * i_resolution.xy;

    shader_main(FragColor, frag_coord);

}

"""
vertex_code_wgsl = """

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) index: u32) -> Varyings {
    var out: Varyings;
    if (index == u32(0)) {
        out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, 1.0);
    } else if (index == u32(1)) {
        out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        out.uv = vec2<f32>(2.0, 1.0);
    } else {
        out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        out.uv = vec2<f32>(0.0, -1.0);
    }
    return out;

}
"""

builtin_variables_wgsl = """

var<private> i_resolution: vec3<f32>;
var<private> i_mouse: vec4<f32>;
var<private> i_time_delta: f32;
var<private> i_time: f32;
var<private> i_frame: u32;

// TODO: more global variables
// var<private> i_frag_coord: vec2<f32>;

"""

fragment_code_wgsl = """

struct ShadertoyInput {
    mouse: vec4<f32>,
    resolution: vec3<f32>,
    time: f32,
    time_delta: f32,
    frame: u32,
};

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@group(0) @binding(0)
var<uniform> input: ShadertoyInput;

@group(1) @binding(0)
var i_channel0: texture_2d<f32>;
@group(1) @binding(1)
var sampler0: sampler;

@group(2) @binding(0)
var i_channel1: texture_2d<f32>;
@group(2) @binding(1)
var sampler1: sampler;

@group(3) @binding(0)
var i_channel2: texture_2d<f32>;
@group(3) @binding(1)
var sampler2: sampler;

@group(4) @binding(0)
var i_channel3: texture_2d<f32>;
@group(4) @binding(1)
var sampler3: sampler;

@fragment
fn main(in: Varyings) -> @location(0) vec4<f32> {

    i_time = input.time;
    i_resolution = input.resolution;
    i_time_delta = input.time_delta;
    i_mouse = input.mouse;
    i_frame = input.frame;

    var uv = in.uv;
    uv.y = 1.0 - uv.y;
    let frag_coord = uv * i_resolution.xy;

    return shader_main(frag_coord);
}

"""

# shared resources in module
gpu_cache = weakref.WeakValueDictionary()


def _get_device():
    device = gpu_cache.get("device", None)
    if device is None:
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        device = adapter.request_device(required_features=["float32-filterable"])
        gpu_cache["device"] = device
    return device


def _get_channel_layout(view_dimension=wgpu.TextureViewDimension.d2):
    layout = gpu_cache.get("channel_layout_" + view_dimension, None)
    if layout is None:
        device = _get_device()
        layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": view_dimension,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )
        gpu_cache["channel_layout_" + view_dimension] = layout
    return layout


def _get_uniform_input_layout():
    layout = gpu_cache.get("uniform_input_layout", None)
    if layout is None:
        device = _get_device()
        layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        gpu_cache["uniform_input_layout"] = layout
    return layout


class ShadertoyChannel:
    def __init__(self, resource, filter="linear", wrap="repeat") -> None:
        self._device = _get_device()
        if isinstance(resource, wgpu.GPUTexture):
            self._texture = resource
        elif isinstance(resource, wgpu.GPUTextureView):
            self._texture = resource.texture

        self._filter = filter
        self._wrap = wrap
        self._sampler = None

        self._bind_group_layout = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )

        self._bind_group = None

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        if value not in ["nearest", "linear"]:
            raise ValueError("Invalid filter value.")

        if value != self._filter:
            self._filter = value
            self._sampler = None

    @property
    def wrap(self):
        return self._wrap

    @wrap.setter
    def wrap(self, value):
        if value not in ["clamp", "repeat"]:
            raise ValueError("Invalid wrap value.")

        if value != self._wrap:
            self._wrap = value
            self._sampler = None

    @property
    def texture(self):
        return self._texture

    @property
    def sampler(self):
        if self._sampler is None:
            wrap = "repeat" if self.wrap == "repeat" else "clamp-to-edge"
            self._sampler = self._device.create_sampler(
                mag_filter=self.filter,
                min_filter=self.filter,
                mipmap_filter=self.filter,
                address_mode_u=wrap,
                address_mode_v=wrap,
                address_mode_w=wrap,
            )
            self._bind_group = None

        return self._sampler

    @property
    def bind_group_layout(self):
        return self._bind_group_layout

    @property
    def bind_group(self):
        if self._bind_group is None:
            view_dimension = (
                wgpu.TextureViewDimension.d2
            )  # todo: get from texture, we should support cube-texture
            self._bind_group = self._device.create_bind_group(
                layout=_get_channel_layout(view_dimension),
                entries=[
                    {"binding": 0, "resource": self.texture.create_view()},
                    {"binding": 1, "resource": self.sampler},
                ],
            )

        return self._bind_group


class BufferChannel(ShadertoyChannel):
    def __init__(self, resolution, filter="linear", wrap="clamp") -> None:
        self._device = _get_device()
        buffer_texture = self._device.create_texture(
            size=(resolution[0], resolution[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        self._target_texture = self._device.create_texture(
            size=(resolution[0], resolution[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        super().__init__(buffer_texture, filter, wrap)

    @property
    def target_texture(self):
        return self._target_texture

    def resize(self, resolution):
        self._texture = self._device.create_texture(
            size=(resolution[0], resolution[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        self._target_texture = self._device.create_texture(
            size=(resolution[0], resolution[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        self._bind_group = None


class DataChannel(ShadertoyChannel):
    def __init__(self, data, filter="linear", wrap="repeat") -> None:
        self._device = _get_device()
        texture = self._create_texture_from_data(data)
        super().__init__(texture, filter, wrap)

    def _create_texture_from_data(self, data):
        size = data.shape
        if len(size) == 2:
            size = (size[1], size[0], 1)

        if size[2] == 1:
            # grayscale image
            format = wgpu.TextureFormat.r8unorm
            bytes_per_pixel = 1
        elif size[2] == 2:
            # RG image
            format = wgpu.TextureFormat.rg8unorm
            bytes_per_pixel = 2
        elif size[2] == 3:
            # RGB image
            # add alpha channel
            data = np.concatenate(
                [data, np.ones((size[0], size[1], 1), dtype=data.dtype)], axis=2
            )
            format = wgpu.TextureFormat.rgba8unorm
            bytes_per_pixel = 4
        elif size[2] == 4:
            # RGBA image
            format = wgpu.TextureFormat.rgba8unorm
            bytes_per_pixel = 4
        else:
            raise ValueError("Invalid image data shape.")

        texture = self._device.create_texture(
            size=size,
            format=format,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        self._device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": size[0] * bytes_per_pixel, "rows_per_image": size[1]},
            size,
        )

        return texture


DEFAULT_CHANNEL = BufferChannel((1, 1))


class ShaderPass:
    def __init__(
        self,
        shader_code,
        render_target: "BufferChannel | wgpu.GPUCanvasContext",
        channel_0=None,
        channel_1=None,
        channel_2=None,
        channel_3=None,
        flip=False,
    ) -> None:
        self._shader_code = shader_code

        self.render_target = render_target

        self._channel_0 = channel_0 or DEFAULT_CHANNEL
        self._channel_1 = channel_1 or DEFAULT_CHANNEL
        self._channel_2 = channel_2 or DEFAULT_CHANNEL
        self._channel_3 = channel_3 or DEFAULT_CHANNEL

        self._flip = flip

        self._render_pipeline = None

    @property
    def shader_code(self):
        """The shader code to use."""
        return self._shader_code

    @property
    def shader_type(self):
        """The shader type, automatically detected from the shader code, can be "wgsl" or "glsl"."""
        if "fn shader_main" in self.shader_code:
            return "wgsl"
        elif (
            "void shader_main" in self.shader_code
            or "void mainImage" in self.shader_code
        ):
            return "glsl"
        else:
            raise ValueError("Invalid shader code.")

    @property
    def channel_0(self):
        return self._channel_0

    @channel_0.setter
    def channel_0(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_0 = value

    @property
    def channel_1(self):
        return self._channel_1

    @channel_1.setter
    def channel_1(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_1 = value

    @property
    def channel_2(self):
        return self._channel_2

    @channel_2.setter
    def channel_2(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_2 = value

    @property
    def channel_3(self):
        return self._channel_3

    @channel_3.setter
    def channel_3(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_3 = value

    def get_render_pipeline(self):
        device = _get_device()
        if self._render_pipeline is None:
            shader_type = self.shader_type
            if shader_type == "glsl":
                vertex_shader_code = vertex_code_glsl
                frag_shader_code = (
                    builtin_variables_glsl + self.shader_code + fragment_code_glsl
                )
            elif shader_type == "wgsl":
                vertex_shader_code = vertex_code_wgsl
                frag_shader_code = (
                    builtin_variables_wgsl + self.shader_code + fragment_code_wgsl
                )

            if not self._flip:
                frag_shader_code = frag_shader_code.replace("uv.y = 1.0 - uv.y;", "")

            vertex_shader_program = device.create_shader_module(
                label="triangle_vert", code=vertex_shader_code
            )
            frag_shader_program = device.create_shader_module(
                label="triangle_frag", code=frag_shader_code
            )

            uniform_buffer_bind_group_layout = _get_uniform_input_layout()

            bind_group_layouts = [uniform_buffer_bind_group_layout]

            for channel in [
                self._channel_0,
                self._channel_1,
                self._channel_2,
                self._channel_3,
            ]:
                bind_group_layouts.append(channel._bind_group_layout)

            self._render_pipeline = device.create_render_pipeline(
                layout=device.create_pipeline_layout(
                    bind_group_layouts=bind_group_layouts
                ),
                vertex={
                    "module": vertex_shader_program,
                    "entry_point": "main",
                    "buffers": [],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": wgpu.FrontFace.ccw,
                    "cull_mode": wgpu.CullMode.none,
                },
                depth_stencil=None,
                multisample=None,
                fragment={
                    "module": frag_shader_program,
                    "entry_point": "main",
                    "targets": [
                        {
                            "format": self.target_format,
                        },
                    ],
                },
            )

        return self._render_pipeline

    @property
    def render_target(self) -> "BufferChannel | wgpu.GPUCanvasContext":
        if isinstance(self._render_target, WgpuCanvas):
            return self._render_target.get_context()

        return self._render_target

    @render_target.setter
    def render_target(self, value):
        assert isinstance(value, (BufferChannel, WgpuCanvas, wgpu.GPUCanvasContext))

        if isinstance(value, WgpuCanvas):
            value = value.get_context()

        self._render_target = value
        self._render_pipeline = None

    @property
    def target_format(self):
        if isinstance(self.render_target, wgpu.GPUCanvasContext):
            return wgpu.TextureFormat.bgra8unorm  # todo: get from canvas context
        else:  # BufferChannel
            return self.render_target.target_texture.format

    def draw_frame(self, command_encoder, uniform_bind_group):
        renter_target = self.render_target

        if isinstance(renter_target, wgpu.GPUCanvasContext):
            target_texture = renter_target.get_current_texture()
        elif isinstance(renter_target, BufferChannel):
            target_texture = renter_target.target_texture
        else:
            print(renter_target)
            raise ValueError("Invalid render target.")

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self.get_render_pipeline())
        render_pass.set_bind_group(0, uniform_bind_group, [], 0, 99)
        for i, channel in enumerate(
            [self._channel_0, self._channel_1, self._channel_2, self._channel_3]
        ):
            render_pass.set_bind_group(i + 1, channel.bind_group, [], 0, 99)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()

        if isinstance(renter_target, BufferChannel):
            command_encoder.copy_texture_to_texture(
                {
                    "texture": renter_target.target_texture,
                    "mip_level": 0,
                    "origin": (0, 0, 0),
                },
                {
                    "texture": renter_target.texture,
                    "mip_level": 0,
                    "origin": (0, 0, 0),
                },
                renter_target.target_texture.size,
            )


class Shadertoy:
    def __init__(
        self,
        main_code,
        common_code="",
        buffer_a_code=None,
        buffer_b_code=None,
        buffer_c_code=None,
        buffer_d_code=None,
        resolution=(800, 450),
    ) -> None:

        self._uniform_data = np.zeros(
            (),
            dtype=[
                ("mouse", "float32", (4)),
                ("resolution", "float32", (3)),
                ("time", "float32"),
                ("time_delta", "float32"),
                ("frame", "uint32"),
                ("__padding", "uint32", (2)),  # padding to 48 bytes
            ],
        )
        self._uniform_data["resolution"] = resolution + (1,)

        self._canvas = WgpuCanvas(title="Shadertoy", size=resolution, max_fps=60)
        self._device = _get_device()
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
            layout=_get_uniform_input_layout(),
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

        if buffer_a_code is not None:
            buffer_a = BufferChannel(resolution)
            self._buffer_a_pass = ShaderPass(
                common_code + buffer_a_code, render_target=buffer_a
            )
        if buffer_b_code is not None:
            buffer_b = BufferChannel(resolution)
            self._buffer_b_pass = ShaderPass(
                common_code + buffer_b_code, render_target=buffer_b
            )
        if buffer_c_code is not None:
            buffer_c = BufferChannel(resolution)
            self._buffer_c_pass = ShaderPass(
                common_code + buffer_c_code, render_target=buffer_c
            )
        if buffer_d_code is not None:
            buffer_d = BufferChannel(resolution)
            self._buffer_d_pass = ShaderPass(
                common_code + buffer_d_code, render_target=buffer_d
            )

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
        self._canvas.request_draw(self._draw_frame)
        run()


if __name__ == "__main__":
    main_code = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    let uv = frag_coord / i_resolution.xy;

    if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }else{
        return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
    }

}
"""
    shader = Shadertoy(main_code)
    shader.show()