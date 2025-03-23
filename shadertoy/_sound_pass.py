import wgpu
from types import NoneType
import numpy as np
from ._channel import ShadertoyChannel, DEFAULT_CHANNEL
from ._pass import ShaderPass
from ._shared import get_device, get_audio_buffer_layout
from ._audio import _AudioPlayer


builtin_variables_wgsl = """
const i_sample_rate: f32 = 44100.0;

@group(0) @binding(0)
var<storage> audio_buffer : array<vec2f, 44100 * 180>;

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
"""

compute_code_wgsl = """
@compute
@workgroup_size(1)  // todo: greater workgroup size
fn c_main(@builtin(global_invocation_id) in: vec3<u32>) {
    let samp = 44100 * in.y + in.x;
    let time = f32(samp) / 44100.0;
    let sampele_value = main_sound(samp, time);
    audio_buffer[samp] = sampele_value;
}
"""

builtin_variables_glsl = """
#version 450 core

const float iSampleRate = 44100.0;

layout(set = 0, binding = 0) buffer AudioBuffer {
    vec2 audio_buffer[44100 * 180];
};

#define iChannel0 sampler2D(i_channel0, sampler0)
#define iChannel1 sampler2D(i_channel1, sampler1)
#define iChannel2 sampler2D(i_channel2, sampler2)
#define iChannel3 sampler2D(i_channel3, sampler3)

#define mainSound main_sound

layout(set = 1, binding = 0) uniform texture2D i_channel0;
layout(set = 1, binding = 1) uniform sampler sampler0;

layout(set = 2, binding = 0) uniform texture2D i_channel1;
layout(set = 2, binding = 1) uniform sampler sampler1;

layout(set = 3, binding = 0) uniform texture2D i_channel2;
layout(set = 3, binding = 1) uniform sampler sampler2;

layout(set = 4, binding = 0) uniform texture2D i_channel3;
layout(set = 4, binding = 1) uniform sampler sampler3;
"""

compute_code_glsl = """
void main() {
    uint samp = 44100 * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x;
    float time = float(samp) / 44100.0;
    vec2 sample_value = main_sound(int(samp), time);
    audio_buffer[samp] = sample_value;
}
"""

class SoundPass:
    def __init__(
        self,
        shader_code,
        channel_0=None,
        channel_1=None,
        channel_2=None,
        channel_3=None,
    ) -> None:
        self._shader_code = shader_code

        self._channel_0 = channel_0 or DEFAULT_CHANNEL
        self._channel_1 = channel_1 or DEFAULT_CHANNEL
        self._channel_2 = channel_2 or DEFAULT_CHANNEL
        self._channel_3 = channel_3 or DEFAULT_CHANNEL

        self._pipeline = None

        device = get_device()
        # Create a buffer to store the audio data
        # 44100 samples per second * 180 seconds * 2 channels * 4 bytes per sample
        self._audio_buffer = device.create_buffer(
            size=44100 * 180 * 2 * 4, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.STORAGE
        )

        self._audio_data = None

    @property
    def shader_code(self):
        """The shader code to use."""
        return self._shader_code

    @property
    def shader_type(self):
        """The shader type, automatically detected from the shader code, can be "wgsl" or "glsl"."""
        if "fn main_sound" in self.shader_code:
            return "wgsl"
        elif (
            "vec2 main_sound" in self.shader_code
            or "vec2 mainSound" in self.shader_code
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

    def get_pipeline(self):
        device = get_device()
        if self._pipeline is None:
            shader_type = self.shader_type
            if shader_type == "glsl":
                compute_shader_code = (
                    builtin_variables_glsl + self.shader_code + compute_code_glsl
                )
            elif shader_type == "wgsl":
                compute_shader_code = (
                    builtin_variables_wgsl + self.shader_code + compute_code_wgsl
                )

            compute_shader_program = device.create_shader_module(
                label="compute", code=compute_shader_code
            )

            audio_buffer_bind_group_layout = get_audio_buffer_layout()

            bind_group_layouts = [audio_buffer_bind_group_layout]

            for channel in [
                self.channel_0,
                self.channel_1,
                self.channel_2,
                self.channel_3,
            ]:
                bind_group_layouts.append(channel.bind_group_layout)

            self._pipeline = device.create_compute_pipeline(
                layout=device.create_pipeline_layout(
                    bind_group_layouts=bind_group_layouts
                ),
                compute={
                    "module": compute_shader_program,
                    "entry_point": "main",
                },
            )

        return self._pipeline
    
    def get_audio_buffer_bind_group(self):
        if self._audio_buffer_bind_group is None:
            device = get_device()
            self._audio_buffer_bind_group = device.create_bind_group(
                layout=get_audio_buffer_layout(),
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._audio_buffer,
                            "offset": 0,
                            "size": self._audio_buffer.size,
                        },
                    }
                ],
            )
        
        return self._audio_buffer_bind_group


    def do_compute(self, command_encoder: "wgpu.GPUCommandEncoder"):
        audio_buffer_bind_group = getattr(self._audio_buffer, "_bind_group", None)
        if audio_buffer_bind_group is None:
            device = get_device()
            audio_buffer_bind_group = device.create_bind_group(
                layout=get_audio_buffer_layout(),
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._audio_buffer,
                            "offset": 0,
                            "size": self._audio_buffer.size,
                        },
                    }
                ],
            )
            self._audio_buffer._bind_group = audio_buffer_bind_group

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.get_pipeline())
        compute_pass.set_bind_group(0, audio_buffer_bind_group)
        for i, channel in enumerate(
            [self.channel_0, self.channel_1, self.channel_2, self.channel_3]
        ):
            channel.update()
            compute_pass.set_bind_group(i + 1, channel.bind_group)
        
        compute_pass.dispatch_workgroups(44100, 180)
        compute_pass.end()

    def get_audio_data(self):
        if self._audio_data is None:
            device = get_device()
            command_encoder = device.create_command_encoder()
            self.do_compute(command_encoder)
            device.queue.submit([command_encoder.finish()])
            data = device.queue.read_buffer(self._audio_buffer, 0, 44100 * 180 * 2 * 4)
            self._audio_data = np.frombuffer(data, np.float32).reshape(-1, 2)
        return self._audio_data
    
    def play(self):
        audio_data = self.get_audio_data()
        audio_player = _AudioPlayer(audio_data)
        audio_player.play()