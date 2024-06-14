# Shadertoy
This is a Shadertoy implementation based on [wgpu-py](https://github.com/pygfx/wgpu-py).

## Introduction
This project provides a "screen pixel shader programming interface" similar to [Shadertoy](https://www.shadertoy.com/), enabling you to easily research, build, or test shaders using WGSL via WGPU.

It supports both WGSL and GLSL shaders. You can almost directly copy code from the [Shadertoy](https://www.shadertoy.com/) website and run it.

## Installation
To install the package, use the following command:
```bash
pip install https://github.com/panxinmiao/shadertoy/archive/main.zip
```

For development or installation from the source code:
```bash
git clone https://github.com/panxinmiao/shadertoy.git
cd shadertoy
pip install -e .
```

You probably also want to install `glfw` (or other gui backends).
```bash
pip install glfw
```
Please check the installation instructions of [wgpu-py](https://github.com/pygfx/wgpu-py) for details.

If you want to use the Audio features (`AudioChannel` or `SoundPass`), you also need to install the `sounddevice` and `soundfile` packages:
```bash
pip install sounddevice
pip install soundfile
```

## Usage
The `Shadertoy` class takes a shader code string as input, and you can use the `show` method to display the shader in a window.

### Basic Example:
```python
from shadertoy import Shadertoy

main_code = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    let uv = frag_coord / i_resolution.xy;

    if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
    }
}
"""
shader = Shadertoy(main_code)
shader.show()
```

### Using Channel Input:
```python
from shadertoy import Shadertoy, DataChannel
import imageio.v3 as iio

# Load an image as a numpy array
noise_img = iio.imread("noise.png")

shader = Shadertoy(main_code)
shader.main_pass.channel_0 = DataChannel(noise_img)
shader.show()
```

### Using AudioChannel:
Note: This requires the `sounddevice` and `soundfile` packages.
```python
from shadertoy import Shadertoy
from shadertoy.audio import AudioChannel

shader = Shadertoy(main_code)
audio_channel = AudioChannel("audio.mp3")
shader.main_pass.channel_0 = audio_channel
audio_channel.play()  # Start playing the audio
shader.show()
```

### Multiple Passes:
You can provide shader code for each pass, including sound passes. Configure the input channel of each pass, and you can set one pass as the channel for another pass, or even for itself.

```python
from shadertoy import Shadertoy, DataChannel
import imageio.v3 as iio

shader = Shadertoy(
    main_code,
    common_code=common_code,
    buffer_a_code=buffer_a_code,
    buffer_b_code=buffer_b_code,
    sound_code=sound_code,  # The sound pass requires the `sounddevice` package
)

# Load an image as a numpy array
noise_img = iio.imread("noise.png")

# Configure channels for each pass
shader.buffer_a_pass.channel_0 = DataChannel(noise_img) # channel_0 is a texture
shader.buffer_a_pass.channel_1 = shader.buffer_a_pass  # channel_1 is the pass itself

shader.buffer_b_pass.channel_0 = shader.buffer_a_pass  # channel_0 is another pass

shader.main_pass.channel_0 = shader.buffer_b_pass
shader.main_pass.channel_1 = shader.buffer_a_pass
shader.show()
```

For more examples, please refer to the [examples](https://github.com/panxinmiao/shadertoy/tree/main/examples) directory.