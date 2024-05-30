import wgpu
import numpy as np
from ._shared import get_device, get_channel_layout

class ShadertoyChannel:
    def __init__(self, resource, filter="linear", wrap="repeat") -> None:
        self._device = get_device()
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
                layout=get_channel_layout(view_dimension),
                entries=[
                    {"binding": 0, "resource": self.texture.create_view()},
                    {"binding": 1, "resource": self.sampler},
                ],
            )

        return self._bind_group


class BufferChannel(ShadertoyChannel):
    def __init__(self, resolution, filter="linear", wrap="clamp") -> None:
        self._device = get_device()
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
        self._device = get_device()
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