import weakref
import wgpu

# shared resources in module
_gpu_cache = weakref.WeakValueDictionary()


def get_device():
    device = _gpu_cache.get("device", None)
    if device is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        device = adapter.request_device_sync(required_features=["float32-filterable"])
        _gpu_cache["device"] = device
    return device


def get_channel_layout(view_dimension=wgpu.TextureViewDimension.d2):
    key = "channel_layout_" + view_dimension
    layout = _gpu_cache.get(key, None)
    if layout is None:
        device = get_device()
        layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": view_dimension,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.COMPUTE,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )
        _gpu_cache[key] = layout
    return layout


def get_uniform_input_layout():
    layout = _gpu_cache.get("uniform_input_layout", None)
    if layout is None:
        device = get_device()
        layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                }
            ]
        )
        _gpu_cache["uniform_input_layout"] = layout
    return layout

def get_sampler(filter="linear", wrap="repeat"):
    assert filter in ["nearest", "linear"]
    assert wrap in ["repeat", "clamp"]
    sampler = _gpu_cache.get("sampler_" + filter + "_" + wrap, None)
    if sampler is None:
        device = get_device()
        wrap = "repeat" if wrap == "repeat" else "clamp-to-edge"
        sampler = device.create_sampler(
            mag_filter=filter,
            min_filter=filter,
            mipmap_filter=filter,
            address_mode_u=wrap,
            address_mode_v=wrap,
            address_mode_w=wrap,
        )
        _gpu_cache["sampler_" + filter + "_" + wrap] = sampler
    return sampler

def get_audio_buffer_layout():
    layout = _gpu_cache.get("audio_buffer_layout", None)
    if layout is None:
        device = get_device()
        layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": wgpu.BufferBindingType.storage,
                    },
                }
            ]
        )
        _gpu_cache["audio_buffer_layout"] = layout
    return layout