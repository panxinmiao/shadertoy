main_code="""
void mainImage( out vec4 o, vec2 i ) { 

// --- color version (base = 110 chars)
    o = step(texture(iChannel0, i/8.).r, texture(iChannel1,i/iResolution.xy));


}

"""


from shadertoy import Shadertoy, TextureChannel, VideoChannel
from pathlib import Path

if __name__ == "__main__":
    shader = Shadertoy(main_code)
    shader.main_pass.channel_0 = TextureChannel(Path(__file__).parent / "media"/"bayer.png", filter="nearest")
    shader.main_pass.channel_1 = VideoChannel(Path(__file__).parent / "media"/"britney.webm", vflip=True)
    shader.show()