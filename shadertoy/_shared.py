import weakref
import wgpu

# shared resources in module
_gpu_cache = weakref.WeakValueDictionary()


def get_device():
    device = _gpu_cache.get("device", None)
    if device is None:
        adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        device = adapter.request_device(required_features=["float32-filterable"])
        _gpu_cache["device"] = device
    return device


def get_channel_layout(view_dimension=wgpu.TextureViewDimension.d2):
    layout = _gpu_cache.get("channel_layout_" + view_dimension, None)
    if layout is None:
        device = get_device()
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
        _gpu_cache["channel_layout_" + view_dimension] = layout
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
