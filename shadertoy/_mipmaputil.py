import math
import wgpu
from ._shared import get_device


class MipmapsUtil:
    def __init__(self) -> None:
        self.device = get_device()

        mipmap_vertex_source = """
struct VarysStruct {
    @builtin( position ) Position: vec4<f32>,
    @location( 0 ) vTex : vec2<f32>
};
@vertex
fn main( @builtin( vertex_index ) vertexIndex : u32 ) -> VarysStruct {
    var Varys : VarysStruct;
    var pos = array< vec2<f32>, 4 >(
        vec2<f32>( -1.0,  1.0 ),
        vec2<f32>(  1.0,  1.0 ),
        vec2<f32>( -1.0, -1.0 ),
        vec2<f32>(  1.0, -1.0 )
    );
    var tex = array< vec2<f32>, 4 >(
        vec2<f32>( 0.0, 0.0 ),
        vec2<f32>( 1.0, 0.0 ),
        vec2<f32>( 0.0, 1.0 ),
        vec2<f32>( 1.0, 1.0 )
    );
    Varys.vTex = tex[ vertexIndex ];
    Varys.Position = vec4<f32>( pos[ vertexIndex ], 0.0, 1.0 );
    return Varys;
}
"""

        mipmap_fragment_source = """
@group( 0 ) @binding( 0 )
var imgSampler : sampler;
@group( 0 ) @binding( 1 )
var img : texture_2d<f32>;
@fragment
fn main( @location( 0 ) vTex : vec2<f32> ) -> @location( 0 ) vec4<f32> {
    return textureSample( img, imgSampler, vTex );
}
"""
        self.sampler = self.device.create_sampler(min_filter="linear")

        # We'll need a new pipeline for every texture format used.
        self.pipelines = {}

        self.mipmap_vertex_shader_module = self.device.create_shader_module(
            code=mipmap_vertex_source
        )

        self.mipmap_fragment_shader_module = self.device.create_shader_module(
            code=mipmap_fragment_source
        )

    def _get_mipmap_pipeline(self, format) -> "wgpu.GPURenderPipeline":

        pipeline = self.pipelines.get(format, None)

        if pipeline is None:

            pipeline = self.device.create_render_pipeline(
                layout=self._create_pipeline_layout(),
                vertex={
                    "module": self.mipmap_vertex_shader_module,
                    "entry_point": "main",
                    "buffers": [],
                },
                fragment={
                    "module": self.mipmap_fragment_shader_module,
                    "entry_point": "main",
                    "targets": [{"format": format}],
                },
                primitive={
                    "topology": "triangle-strip",
                    "strip_index_format": "uint32",
                },
            )

            self.pipelines[format] = pipeline

        return pipeline

    def _create_pipeline_layout(self):
        bind_group_layouts = []

        entries = []
        entries.append(
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            }
        )

        entries.append(
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"multisampled": False},
            }
        )

        bind_group_layout = self.device.create_bind_group_layout(entries=entries)
        bind_group_layouts.append(bind_group_layout)
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=bind_group_layouts
        )
        return pipeline_layout

    def generate_mipmaps(
        self,
        texture_gpu: "wgpu.GPUTexture",
        base_array_layer=0,
    ):
        format = texture_gpu.format
        mip_level_count = texture_gpu.mip_level_count
        pipeline = self._get_mipmap_pipeline(format)

        command_encoder: "wgpu.GPUCommandEncoder" = self.device.create_command_encoder()
        bind_group_layout = pipeline.get_bind_group_layout(0)

        src_view = texture_gpu.create_view(
            base_mip_level=0,
            mip_level_count=1,
            dimension="2d",
            base_array_layer=base_array_layer,
        )

        for i in range(1, mip_level_count):

            dst_view = texture_gpu.create_view(
                base_mip_level=i,
                mip_level_count=1,
                dimension="2d",
                base_array_layer=base_array_layer,
            )

            pass_encoder: "wgpu.GPURenderPassEncoder" = (
                command_encoder.begin_render_pass(
                    color_attachments=[
                        {
                            "view": dst_view,
                            "load_op": wgpu.LoadOp.clear,
                            "store_op": wgpu.StoreOp.store,
                            "clear_value": [0, 0, 0, 0],
                        }
                    ]
                )
            )

            bind_group = self.device.create_bind_group(
                layout=bind_group_layout,
                entries=[
                    {"binding": 0, "resource": self.sampler},
                    {"binding": 1, "resource": src_view},
                ],
            )

            pass_encoder.set_pipeline(pipeline)
            pass_encoder.set_bind_group(0, bind_group, [], 0, 99)
            pass_encoder.draw(4, 1, 0, 0)
            pass_encoder.end()

            src_view = dst_view

        self.device.queue.submit([command_encoder.finish()])


_the_mipmap_util = None


def get_mipmaps_util():
    global _the_mipmap_util
    if not _the_mipmap_util:
        _the_mipmap_util = MipmapsUtil()
    return _the_mipmap_util

def generate_mipmaps(texture: wgpu.GPUTexture):
    mipmap_util = get_mipmaps_util()
        # a cube or 2d_array texture, generate mipmaps for each individual layer
    if texture.dimension == wgpu.TextureDimension.d2 and texture.depth_or_array_layers > 1:
        for i in range(texture.depth_or_array_layers):
            mipmap_util.generate_mipmaps(texture, base_array_layer=i)
    else:
        mipmap_util.generate_mipmaps(texture, 0)


def get_mip_level_count(size):
    width, height = size[0], size[1]
    return math.floor(math.log2(max(width, height))) + 1