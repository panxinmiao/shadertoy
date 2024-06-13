from shadertoy import Shadertoy, DataChannel

main_code = """
//// [2TC 15] Mystery Mountains.
// David Hoskins.

// Add texture layers of differing frequencies and magnitudes...
#define F +texture(iChannel0,.3+p.xz*s/3e3)/(s+=s) 

void mainImage( out vec4 c, vec2 w )
{
    vec4 p=vec4(w/iResolution.xy,1,1)-.5,d=p,t;
    p.z += iTime*20.;d.y-=.4;
    
    for(float i=1.5;i>0.;i-=.002)
    {
        float s=.5;
        t = F F F F F F;
        c =1.+d.x-t*i; c.z-=.1;
        if(t.x>p.y*.007+1.3)break;
        p += d;
    }
}
"""

import imageio.v3 as iio
from pathlib import Path

if __name__ == "__main__":
    shader = Shadertoy(main_code)
    img = iio.imread(Path(__file__).parent / "media"/"stars.jpg")
    shader.main_pass.channel_0 = DataChannel(img)
    shader.show()
