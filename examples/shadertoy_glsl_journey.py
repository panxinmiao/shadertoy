from shadertoy import Shadertoy, DataChannel

common_code = """
#define GLOW_SAMPLES 40
#define GLOW_DISTANCE 0.3
#define GLOW_POW .8
#define GLOW_OPACITY .4

#define sat(a) clamp(a,0., 1.)
"""

main_code = """
// This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0
// Unported License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ 
// or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
// =========================================================================================================

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;

    const int steps = GLOW_SAMPLES;
    vec3 col = vec3(0.);
    
    for (int i = 0; i< steps; ++i)
    {
        float f = float(i)/float(steps);
        f = (f -.5)*2.;
        float factor = GLOW_DISTANCE;
        vec2 nuv = uv+vec2(0.,f*factor);
        if (nuv.y > 0. && nuv.y < 1.)
            col += texture(iChannel0, uv+vec2(0.,f*factor)).xyz/float(steps);
    }
    
    vec3 rgb = texture(iChannel1, uv).xyz+GLOW_OPACITY*pow(col, vec3(GLOW_POW));
    rgb = pow(rgb, vec3(1.7));
    vec2 cuv = (fragCoord-.5*iResolution.xy)/iResolution.xx;
    rgb *= 1.-sat(length(cuv*2.)-.5);
    fragColor = vec4(rgb,1.0);

}
"""

buffer_a_code = """

#define GROUND_MAT 0.
#define PILLAR_MAT 1.

mat2 r2d(float a) {float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

// Stolen from 0b5vr here https://www.shadertoy.com/view/ss3SD8
float hash11(float p)
{
    return (fract(sin((p)*114.514)*1919.810));
}
float seed;
float rand()
{
    seed++;
    return hash11(seed);
}
float fixedseed;
float fixedrand()
{
    fixedseed++;
    return hash11(fixedseed);
}

vec2 _min(vec2 a, vec2 b)
{
    if (a.x < b.x)
        return a;
    return b;
}

float _cube(vec3 p, vec3 s)
{
    vec3 l = abs(p)-s;
    return max(l.x, max(l.y,l.z));
}

float _pillar(vec3 p)
{
    float acc = _cube(p, vec3(.4,10.,.4));
    acc = min(acc, _cube(p-vec3(0.,9.2,0.), vec3(.42,10.,.42)));
    float vertholth = .1;
    acc = max(acc, -_cube(p, vec3(1.,10.,vertholth)));
    acc = max(acc, -_cube(p, vec3(vertholth,10.,1.)));
    acc = min(acc, _cube(p, vec3(.1,10.,.1)));
    
    acc = min(acc, _cube(p-vec3(0.,9.75,0.), vec3(.44,10.,.44)));
    
    // Upper part
    acc = min(acc, _cube(p+vec3(0.,12.,0.), vec3(.5,10.,.5)));
    float w1 = .3;
    float l1 = .55;
    acc = min(acc, _cube(p+vec3(0.,2.5,0.), vec3(w1,w1,l1)));
    acc = min(acc, _cube(p+vec3(0.,2.5,0.), vec3(l1,w1,w1)));
    
    float w2 = .2;
    float l2 = .6;
    acc = min(acc, _cube(p+vec3(0.,2.5,0.), vec3(w2,w2,l2)));
    acc = min(acc, _cube(p+vec3(0.,2.5,0.), vec3(l2,w2,w2)));
    acc = acc-.1*(texture(iChannel0, p.xy*.2).x*.01-texture(iChannel0, p.zy*.2).x*.01);
    return acc;
}

vec2 map(vec3 p)
{
    vec3 op = p;
    vec2 acc = vec2(1000.,-1.);
    
    float ground = -p.y
    -sin(p.z*10.-p.x*1.)*.01*sin(p.x*.5+iTime*.5)*sat(sin(p.z*.5))
    -sin((p.z+p.x*.2)*10.)*.01*sat(sin(p.x*1.5+p.z))
    -.5*(sin(p.z*.75+p.x)*1.5-sin(p.x*.5+iTime*.5)*sin(p.z*5.+p.x*10.)*.05)*pow(sat(abs(p.x)/10.),1.);
    acc = _min(acc, vec2(ground, GROUND_MAT));
   
   
    p.x = abs(p.x);
    p.x -= 2.;
    float rep = 3.;
    
    p.z = mod(p.z+rep*.5, rep)-rep*.5;
    acc = _min(acc, vec2(_pillar(p), PILLAR_MAT));
    
    op -= vec3(-1.,-2.5,0.);
    vec3 p1 = op-vec3(sin(iTime*3.)+.5*sin(iTime), sin(iTime*1.7)*.5,sin(iTime*3.7));    
    acc = _min(acc, vec2(length(p1)-.05, 2.));
    vec3 p2 = op-vec3(sin(-iTime*3.)+.5*sin(-iTime), sin(iTime*1.7)*.5,sin(-iTime*1.7));    
    acc = _min(acc, vec2(length(p2)-.01, 2.));
    
    return acc;
}

vec3 getCam(vec3 rd, vec2 uv)
{
    float fov = 2.;
    vec3 r = normalize(cross(rd, vec3(0.,1.,0.)));
    vec3 u = normalize(cross(rd, r));
    return normalize(rd+(r*uv.x+u*uv.y)*fov);
}

vec3 getNorm(vec3 p, float d)
{
    vec2 e = vec2(0.001,0.);
    return normalize(vec3(d)-vec3(map(p-e.xyy).x, map(p-e.yxy).x, map(p-e.yyx).x));
}
vec3 accCol;
vec3 trace(vec3 ro, vec3 rd, int steps)
{
    accCol = vec3(0.);
    vec3 p = ro;
    for (int i = 0; i < steps; ++i)
    {
        vec2 res = map(p);
        if (res.x<0.001)
            return vec3(res.x, distance(p, ro), res.y);
        vec3 pl = p;
        pl.xy *= r2d(-.25);
        if (length(pl.xz-vec2(-2.5,1.)) < 2.)
            accCol += .015*vec3(1.000,0.733,0.361)*(1.-sat(res.x/1.51))*sat(-p.y-1.);
        p+= rd*res.x*.25;
    }
    return vec3(-1.);
}

vec3 getMat(vec3 res, vec3 rd, vec3 p, vec3 n)
{
    vec3 col = vec3(0.);
    vec3 lpos = vec3(-2.,15.,15.);
    vec3 ldir = p-lpos;
    vec3 h = normalize(rd+ldir);
    float ndoth = dot(n,h);
    if (res.z == GROUND_MAT)
    {
        vec3 sandn = (vec3(fixedrand(), fixedrand(),fixedrand())-.5)*2.;
        float ndoth2 = dot(normalize(n+sandn*.1),h);
        col = vec3(0.118,0.322,0.243)*pow(sat(ndoth2),10.)*2.;
        col += vec3(0.118,0.322,0.243)*pow(sat(ndoth2),2.)*.5;
        col += vec3(0.459,0.686,0.376)*pow(sat(ndoth2),20.)*15.*(1.-sat(res.y/10.));
        col *= 3.;
    }
    else if (res.z == PILLAR_MAT)
    {
        float pattern = texture(iChannel0, p.xy*.4).x-texture(iChannel0, p.zy*.4).x;
        col = vec3(.1)*sat(pattern+.75)*.25;
        col += 1.2*vec3(.1,.23,.34)*(1.-sat(abs(p.y+1.)*.5))*.5*pow(sat(ndoth),.25);
        vec3 pl = p-vec3(-1.,0.,0.);
        col += 1.5*vec3(1.000,0.733,0.361)*(1.-sat(length(pl.xz)-5.))*.35*sat(1.2+ndoth);
    }
    else
        col = n*.5+.5;

    return col;
}

vec3 rdr(vec2 uv)
{
    vec3 col = vec3(0.);
    
    vec3 dof = (vec3(rand(), rand(), rand())-.5)*.1*sat(length(uv)*.5);
    vec3 ro = vec3(1.4-.1*sin(iTime*.25),-.2,-3.)+dof;
    vec3 ta = vec3(0.,-1.5,0.);
    vec3 rd = normalize(ta-ro);
    
    rd = getCam(rd, uv)-dof;
    vec3 res = trace(ro, rd, 1024);
    float maxDist = 100.;
    float dist = maxDist;
    vec3 halolight = vec3(0.);
    if (res.y > 0.)
    {
        halolight = accCol;
        dist = res.y;
        vec3 p = ro+rd*res.y;
        fixedseed = texture(iChannel0,p.xz*.005).x;
        vec3 n = getNorm(p, res.x);
        col = getMat(res, rd, p, n);
        vec3 refl = normalize(reflect(rd,n)+.5*(vec3(rand(),rand(),rand())-.5));
        vec3 resrefl = trace(p+n*0.01, refl, 128);
        if (resrefl.y> 0.)
        {
            vec3 prefl = p*refl*resrefl.y;
            vec3 nrefl = getNorm(prefl, resrefl.x);
            col += getMat(resrefl, rd, prefl, nrefl)*(res.z == GROUND_MAT ? 2.5 : .2);
        }
    }
    
    vec3 fogcol = vec3(.1,.34,.21)*.5*sat(uv.y+.75);//mix(vec3(.1,.34,.21)*.5, vec3(0.784,0.784,0.639)*.1, sat((dist-20.)*.05));
    col = mix(col, fogcol, pow(sat(dist/maxDist), .25));
    col += halolight*1.2;
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.xx;
    seed+= texture(iChannel0, uv).x;
    seed += iTime;
    
    vec3 col = rdr(uv);
    col = mix(col, texture(iChannel1, fragCoord/iResolution.xy).xyz, .85);
    fragColor = vec4(col,1.0);
}
"""


buffer_b_code = """
// This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0
// Unported License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ 
// or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
// =========================================================================================================

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;

    const int steps = GLOW_SAMPLES;
    vec3 col = vec3(0.);
    
    for (int i = 0; i< steps; ++i)
    {
        float f = float(i)/float(steps);
        f = (f -.5)*2.;
        float factor = GLOW_DISTANCE;
        vec2 nuv = uv+vec2(f*factor, 0.);
        if (nuv.x > 0. && nuv.x < 1.)
          col += texture(iChannel0, uv+vec2(f*factor,0.)).xyz/float(steps);
    }
    fragColor = vec4(col,1.0);
}
"""

import imageio.v3 as iio
from pathlib import Path

if __name__ == "__main__":
    shader = Shadertoy(
        main_code,
        buffer_a_code=buffer_a_code,
        buffer_b_code=buffer_b_code,
        common_code=common_code,
    )
    noise_img = iio.imread(Path(__file__).parent / "shadertoy_noise.png")
    shader.buffer_a_pass.channel_0 = DataChannel(noise_img)
    shader.buffer_a_pass.channel_1 = shader.buffer_a_pass
    shader.buffer_b_pass.channel_0 = shader.buffer_a_pass
    shader.main_pass.channel_0 = shader.buffer_b_pass
    shader.main_pass.channel_1 = shader.buffer_a_pass
    shader.show()
