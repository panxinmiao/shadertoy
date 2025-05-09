# Shadertoy
This is a Shadertoy implementation based on [wgpu-py](https://github.com/pygfx/wgpu-py).

## Introduction
This project provides a "screen pixel shader programming interface" similar to [Shadertoy](https://www.shadertoy.com/), enabling you to easily research, build, or test shaders using WGSL via WGPU([wgpu-py](https://github.com/pygfx/wgpu-py)).

It supports both WGSL and GLSL shaders. You can almost directly copy code from the [Shadertoy](https://www.shadertoy.com/) website and run it.

## Installation
To install the package, use the following command:
```bash
pip install https://github.com/panxinmiao/shadertoy/archive/main.zip
```

For development:
```bash
git clone https://github.com/panxinmiao/shadertoy.git
cd shadertoy
pip install -e .
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

### Channel Input:
You can set the input channel for each pass. 

At present, the following channel types are supported:
- TextureChannel: Load an image as a 2d texture.
- AudioChannel: Load an audio file as a sound texture.
- DataChannel: The data is a numpy array
- BufferChannel: Use for multipass
- CubeTextureChannel: Cube texture.
- VolumeTextureChannel: 3D texture.
- VideoChannel: Load a video file as a video texture. (Requires the `moviepy` package, `pip install moviepy`)

Todo: Add keyboardChannel, WebcamChannel.

```python
from shadertoy import Shadertoy, TextureChannel, DataChannel, AudioChannel

shader = Shadertoy(main_code)
shader.main_pass.channel_0 = TextureChannel("noise.png")
shader.main_pass.channel_1 = AudioChannel("audio.mp3")
shader.main_pass.channel_2 = DataChannel(...)

# to use a VideoChannel, you need to install the moviepy package first
from shadertoy import VideoChannel
shader.main_pass.channel_3 = VideoChannel("video.mp4")
shader.show()
```

### Multipass:
You can provide shader code for each pass, including a sound pass. Configure the input channel of each pass, and you can set one pass as the channel for another pass, or even for itself.

```python
from shadertoy import Shadertoy, DataChannel
import imageio.v3 as iio

shader = Shadertoy(
    main_code,
    common_code=common_code,
    buffer_a_code=buffer_a_code,
    buffer_b_code=buffer_b_code,
    sound_code=sound_code,
)

# Load an image as a numpy array
noise_img = iio.imread("noise.png")

# Configure channels for each pass
shader.buffer_a_pass.channel_0 = DataChannel(noise_img)  # pass_a.channel_0 is a texture
shader.buffer_a_pass.channel_1 = shader.buffer_a_pass  # pass_a.channel_1 is itself
shader.buffer_b_pass.channel_0 = shader.buffer_a_pass  # pass_b.channel_0 is another pass

shader.main_pass.channel_0 = shader.buffer_b_pass
shader.main_pass.channel_1 = shader.buffer_a_pass
shader.show()
```

For more examples, please refer to the [examples](https://github.com/panxinmiao/shadertoy/tree/main/examples) directory.

### Capture Frame or Record Video:
You can use the `snapshot()` method to capture the current frame as a numpy array.

```python
img = shader.snapshot() 
# img is a numpy array with shape (height, width, 4)
imageio.imwrite("output.png", img)
```

If you want to capture a frame at a specific timestamp (or other shader state), you can set the shader state before calling the `snapshot()` method.

```python
shader.set_shader_state(
    time = 10.0,
    time_delta = ...,
    ...,
)

img = shader.snapshot()
```

To record a video, you need to install the `MoviePy` package first.:
```bash
pip install moviepy
```

Then you can use the `to_video` method to record a video.

```python
shader.to_video("output.mp4", duration=50, fps=60, resolution=(1280, 720))
```


### CLI Usage:
You can also run from the command line to display a shader from the website [Shadertoy](https://www.shadertoy.com/) quickly.

To do this, you need to **set the `SHADERTOY_API_KEY` environment variable** to your [Shadertoy API key](https://www.shadertoy.com/howto#q2).
```bash
python -m shadertoy XtlSD7 --resolution 800 450
```
Or, if you have installed as a package, you can run it directly:
```bash
shadertoy XtlSD7
```

You can also record a video use "--output" option (requires the `MoviePy` package):
```bash
shadertoy XtlSD7 --output video.mp4 --duration 50 --fps 60 --resolution 1280 720
```

#### Note:

Not all shaders in the website are accessible with API, depending on the shader author's settings. If you encounter a "shader not found" error, it probably means the shader is not accessible via API. 

Since WGPU and Naga are still in fast development, some shaders may not work as expected. If you encounter any shader validation errors, please check the wgpu [issues](https://github.com/gfx-rs/wgpu/issues), and report them if necessary.

If you find that some examples in the [examples](https://github.com/panxinmiao/shadertoy/tree/main/examples) directory works well but encounter validation errors when loading the same examples directly from API, it's likely because I've made some minor adaptations and adjustments in the examples.