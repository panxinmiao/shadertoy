import wgpu
import numpy as np
from types import NoneType
from wgpu.gui import WgpuCanvasBase
from ._channel import BufferChannel, ShadertoyChannel, DEFAULT_CHANNEL
from ._shared import get_device, get_uniform_input_layout

vertex_code_glsl = """
#version 450 core

layout(location = 0) out vec2 v_uv;

void main(void){
    int index = int(gl_VertexID);
    if (index == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
        v_uv = vec2(0.0, 1.0);
    } else if (index == 1) {
        gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
        v_uv = vec2(2.0, 1.0);
    } else {
        gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
        v_uv = vec2(0.0, -1.0);
    }
}
"""

builtin_variables_glsl = """
#version 450 core

vec3 iResolution;
vec4 iMouse;
float iTime;
float iTimeDelta;
int iFrame;

float iFrameRate;
vec4 iDate;
float iChannelTime[4];
vec3 iChannelResolution[4];

// Shadertoy compatibility, see we can use the same code copied from shadertoy website

#define iChannel0 sampler2D(i_channel0, sampler0)
#define iChannel1 sampler2D(i_channel1, sampler1)
#define iChannel2 sampler2D(i_channel2, sampler2)
#define iChannel3 sampler2D(i_channel3, sampler3)

layout(set = 0, binding = 0) uniform texture2D i_channel0;
layout(set = 0, binding = 1) uniform sampler sampler0;

layout(set = 1, binding = 0) uniform texture2D i_channel1;
layout(set = 1, binding = 1) uniform sampler sampler1;

layout(set = 2, binding = 0) uniform texture2D i_channel2;
layout(set = 2, binding = 1) uniform sampler sampler2;

layout(set = 3, binding = 0) uniform texture2D i_channel3;
layout(set = 3, binding = 1) uniform sampler sampler3;

uniform struct ShadertoyInput {
    vec4 _mouse;
    float _time;
    float _time_delta;
    int _frame;
    float _frame_rate;
    vec4 _date;
    vec3 _resolution;
};

uniform struct PassInput {
    float _channel_time_0;
    float _channel_time_1;
    float _channel_time_2;
    float _channel_time_3;
    vec3 _channel_resolution[4];
};

layout(set = 4, binding = 0) uniform ShadertoyInput input;
layout(set = 5, binding = 0) uniform PassInput pass_input;


"""
fragment_code_glsl = """
layout(location = 0) in vec2 v_uv;
out vec4 FragColor;

void main(){

    iTime = input._time;
    iResolution = input._resolution;
    iTimeDelta = input._time_delta;
    iMouse = input._mouse;
    iFrame = input._frame;
    iDate = input._date;
    iFrameRate = input._frame_rate;

    iChannelTime[0] = pass_input._channel_time_0;
    iChannelTime[1] = pass_input._channel_time_1;
    iChannelTime[2] = pass_input._channel_time_2;
    iChannelTime[3] = pass_input._channel_time_3;
    
    iChannelResolution = pass_input._channel_resolution;

    vec2 uv = v_uv;
    uv.y = 1.0 - uv.y;

    vec2 frag_coord = uv * iResolution.xy;

    mainImage(FragColor, frag_coord);

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

var<private> i_date: vec4<f32>;
var<private> i_frame_rate: f32;
var<private> i_channel_time: array<f32, 4>;
var<private> i_channel_resolution: array<vec3<f32>, 4>;

"""

fragment_code_wgsl = """

struct ShadertoyInput {
    mouse: vec4<f32>,
    time: f32,
    time_delta: f32,
    frame: u32,
    frame_rate: f32,
    date: vec4<f32>,
    resolution: vec3<f32>,
};

struct PassInput {
    channel_time_0: f32,
    channel_time_1: f32,
    channel_time_2: f32,
    channel_time_3: f32,
    channel_resolution: array<vec3<f32>, 4>,
};

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};


@group(0) @binding(0)
var i_channel0: texture_2d<f32>;
@group(0) @binding(1)
var sampler0: sampler;

@group(1) @binding(0)
var i_channel1: texture_2d<f32>;
@group(1) @binding(1)
var sampler1: sampler;

@group(2) @binding(0)
var i_channel2: texture_2d<f32>;
@group(2) @binding(1)
var sampler2: sampler;

@group(3) @binding(0)
var i_channel3: texture_2d<f32>;
@group(3) @binding(1)
var sampler3: sampler;

@group(4) @binding(0)
var<uniform> input: ShadertoyInput;

@group(5) @binding(0)
var<uniform> pass_input: PassInput;

@fragment
fn main(in: Varyings) -> @location(0) vec4<f32> {

    i_time = input.time;
    i_resolution = input.resolution;
    i_time_delta = input.time_delta;
    i_mouse = input.mouse;
    i_frame = input.frame;
    i_date = input.date;
    i_frame_rate = input.frame_rate;

    i_channel_time[0] = pass_input.channel_time_0;
    i_channel_time[1] = pass_input.channel_time_1;
    i_channel_time[2] = pass_input.channel_time_2;
    i_channel_time[3] = pass_input.channel_time_3;
    i_channel_resolution = pass_input.channel_resolution;

    var uv = in.uv;
    uv.y = 1.0 - uv.y;
    let frag_coord = uv * i_resolution.xy;

    return shader_main(frag_coord);
}

"""

class ShaderPass:
    def __init__(
        self,
        shader_code,
        render_target: "BufferChannel | wgpu.GPUCanvasContext" = None,
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

        self._uniform_data = np.zeros(
            (),
            dtype=[
                ("channel_time", "float32", (4)),
                ("channel_resolution", "float32", (4, 4)),
            ],
        )

        device = get_device()
        self._uniform_buffer = device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self._uniform_buffer_bind_group = device.create_bind_group(
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
        return self._channel_0 or DEFAULT_CHANNEL

    @channel_0.setter
    def channel_0(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass, NoneType))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_0 = value

    @property
    def channel_1(self):
        return self._channel_1 or DEFAULT_CHANNEL

    @channel_1.setter
    def channel_1(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass, NoneType))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_1 = value

    @property
    def channel_2(self):
        return self._channel_2 or DEFAULT_CHANNEL

    @channel_2.setter
    def channel_2(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass, NoneType))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_2 = value

    @property
    def channel_3(self):
        return self._channel_3 or DEFAULT_CHANNEL

    @channel_3.setter
    def channel_3(self, value):
        assert isinstance(value, (ShadertoyChannel, ShaderPass, NoneType))
        if isinstance(value, ShaderPass):
            value = value.render_target
        self._channel_3 = value

    def get_render_pipeline(self):
        device = get_device()
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


            bind_group_layouts = [] # shadertoy buffer, pass buffer

            for channel in [
                self.channel_0,
                self.channel_1,
                self.channel_2,
                self.channel_3,
            ]:
                bind_group_layouts.append(channel.bind_group_layout)

            buffer_layout = get_uniform_input_layout()
            bind_group_layouts.append(buffer_layout) # shadertoy buffer layout
            bind_group_layouts.append(buffer_layout) # pass buffer layout

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
        if isinstance(self._render_target, WgpuCanvasBase):
            return self._render_target.get_context()

        return self._render_target

    @render_target.setter
    def render_target(self, value):
        assert isinstance(value, (BufferChannel, WgpuCanvasBase, wgpu.GPUCanvasContext, NoneType))

        if isinstance(value, WgpuCanvasBase):
            value = value.get_context()

        self._render_target = value
        self._render_pipeline = None

    @property
    def target_format(self):
        if isinstance(self.render_target, wgpu.GPUCanvasContext):
            return self.render_target._config["format"]
        else:  # BufferChannel
            return self.render_target.target_texture.format

    def draw_frame(self, command_encoder, shadertoy_uniform_bind_group):
        self._update()

        renter_target = self.render_target

        if isinstance(renter_target, wgpu.GPUCanvasContext):
            target_texture = renter_target.get_current_texture()
        elif isinstance(renter_target, BufferChannel):
            target_texture = renter_target.target_texture
        else:
            raise ValueError("Invalid render target.", renter_target)

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_texture.create_view(usage=wgpu.TextureUsage.RENDER_ATTACHMENT),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self.get_render_pipeline())
        for i, channel in enumerate(
            [self.channel_0, self.channel_1, self.channel_2, self.channel_3]
        ):
            channel.update()
            render_pass.set_bind_group(i, channel.bind_group)

        render_pass.set_bind_group(4, shadertoy_uniform_bind_group)
        render_pass.set_bind_group(5, self._uniform_buffer_bind_group)

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

    def _update(self):
        if self._channel_0:
            self._uniform_data["channel_resolution"][0][:3] = self._channel_0.texture.size
            self._uniform_data["channel_time"][0] = self._channel_0.time

        if self._channel_1:
            self._uniform_data["channel_resolution"][1][:3] = self._channel_1.texture.size
            self._uniform_data["channel_time"][1] = self._channel_1.time

        if self._channel_2:
            self._uniform_data["channel_resolution"][2][:3] = self._channel_2.texture.size
            self._uniform_data["channel_time"][2] = self._channel_2.time

        if self._channel_3:
            self._uniform_data["channel_resolution"][3][:3] = self._channel_3.texture.size
            self._uniform_data["channel_time"][3] = self._channel_3.time

    
    def play(self):
        channels = [self.channel_0, self.channel_1, self.channel_2, self.channel_3]

        for channel in channels:
            if channel and hasattr(channel, "play"):
                channel.play()