import wgpu
import numpy as np
from ._shared import get_device, get_channel_layout, get_sampler
from ._mipmapsutil import get_mip_level_count, generate_mipmaps

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

        self._bind_group = None

    def update(self):
        pass

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
            filter = self.filter
            if filter == "mipmap":
                filter = "linear"
            self._sampler = get_sampler(filter, self.wrap)
            self._bind_group = None

        return self._sampler

    @property
    def bind_group_layout(self):
        view_dimension = (
            wgpu.TextureViewDimension.d2
        )  # todo: get from texture, we should support cube-texture
        return get_channel_layout(view_dimension)
    
    @property
    def bind_group(self):
        if self._bind_group is None:
            if self.filter == "mipmap":
                texture_view = self._texture.create_view()
            else:
                texture_view = self._texture.create_view(
                    base_mip_level=0,
                    mip_level_count=1
                )
            self._bind_group = self._device.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {"binding": 0, "resource": texture_view},
                    {"binding": 1, "resource": self.sampler},
                ],
            )

        return self._bind_group


class BufferChannel(ShadertoyChannel):
    def __init__(self, size, filter="linear", wrap="clamp") -> None:
        self._device = get_device()
        buffer_texture = self._device.create_texture(
            size=(size[0], size[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        self._target_texture = self._device.create_texture(
            size=(size[0], size[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        super().__init__(buffer_texture, filter, wrap)

    @property
    def target_texture(self):
        return self._target_texture

    def resize(self, size):
        self._texture = self._device.create_texture(
            size=(size[0], size[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        self._target_texture = self._device.create_texture(
            size=(size[0], size[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        self._bind_group = None


class DataChannel(ShadertoyChannel):
    def __init__(self, data, filter="linear", wrap="repeat", vflip=False) -> None:
        self._device = get_device()

        self._data = data

        if filter == "mipmap":
            need_mipmaps = True
        else:
            need_mipmaps = False

        self._vflip = vflip

        texture = self._create_texture_from_data(data, need_mipmaps, vflip)
        super().__init__(texture, filter, wrap)

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, value):
        if value not in ["nearest", "linear", "mipmap"]:
            raise ValueError("Invalid filter value.")

        if value != self._filter:
            self._filter = value
            self._sampler = None
        if value == "mipmap" and not self._has_mipmaps:
            self._texture.destroy()
            self._texture = self._create_texture_from_data(self._data, True)

    @property
    def vflip(self):
        return self._vflip
    
    @vflip.setter
    def vflip(self, value):
        if value != self._vflip:
            self._vflip = value
            self._texture.destroy()
            self._texture = self._create_texture_from_data(self._data, self._has_mipmaps, value)
        

    def _create_texture_from_data(self, data, need_mipmaps=False, vflip=False):
        shape = data.shape # NHWC

        if vflip:
            data = np.ascontiguousarray(np.flipud(data))

        if len(shape) == 2: # HW
            shape = shape + (1,)

        if len(shape) == 3: # we assume it's HWC (maybe NHW?）
            shape = (1,) + shape

        data = data.reshape(shape)

        size = (shape[2], shape[1], shape[0])  # width, height, depth

        if shape[3] == 1:
            # grayscale image
            format = wgpu.TextureFormat.r8unorm
            bytes_per_pixel = 1
        elif shape[3] == 2:
            # RG image
            format = wgpu.TextureFormat.rg8unorm
            bytes_per_pixel = 2
        elif shape[3] == 3:
            # RGB image
            # add alpha channel
            data = np.concatenate(
                [data, np.ones(shape[:3] + (1,), dtype=data.dtype)], axis=-1
            )
            format = wgpu.TextureFormat.rgba8unorm
            bytes_per_pixel = 4
        elif shape[3] == 4:
            # RGBA image
            format = wgpu.TextureFormat.rgba8unorm
            bytes_per_pixel = 4
        else:
            raise ValueError("Invalid image data shape.")
        
        if need_mipmaps:
            mipmap_level_count = get_mip_level_count(size)
            usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.RENDER_ATTACHMENT
        else:
            mipmap_level_count = 1
            usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST

        texture = self._device.create_texture(
            size=size,
            format=format,
            usage=usage,
            mip_level_count=mipmap_level_count,
        )

        self._device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"bytes_per_row": size[0] * bytes_per_pixel, "rows_per_image": size[1]},
            size,
        )

        if need_mipmaps:
            generate_mipmaps(texture)
        
        self._has_mipmaps = need_mipmaps

        return texture

DEFAULT_CHANNEL = DataChannel(np.zeros((1,1), dtype=np.uint8))
