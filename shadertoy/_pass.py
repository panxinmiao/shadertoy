import wgpu
from wgpu.gui.auto import WgpuCanvas
from ._channel import BufferChannel, ShadertoyChannel, DEFAULT_CHANNEL
from ._shared import get_device, get_uniform_input_layout

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

            uniform_buffer_bind_group_layout = get_uniform_input_layout()

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