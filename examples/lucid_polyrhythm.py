# https://www.shadertoy.com/view/43BGRV

main_code = """
/*
vec3 lch2srgb(vec3 v)
{
    v = vec3(v.x, v.y * cos(v.z), v.y * sin(v.z)) * mat3(1,   .396338,   .215804,
                                                         1,  -.105561, -.0638542,
                                                         1, -.0894842, - 1.29149);
    
    v = v * v * v * mat3(4.07659   , -3.30717,  .230732,
                         -1.26814  ,  2.60934, -.341134,
                         -.00411222, -.703477,  1.70686);
    return mix(1.055 * pow(v, vec3(1) / 2.4) - .055, v * 12.92, lessThan(v, vec3(.0031308)));
}
*/

vec3 lch2rgb(vec3 v)
{
    v = vec3(v.x, v.y * cos(v.z), v.y * sin(v.z)) * mat3(1,  .396,   .216,
                                                         1, -.106, - .064,
                                                         1, -.09 , -1.292);
    
    v = v * v * v * mat3( 4.077, -3.307,   .231,
                         -1.268,  2.609, - .341,
                         - .004, - .704,  1.707);
    
    return v * v; // approximation
}

vec3 hash(vec3 p3)
{
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yxz+33.33);
    return fract((p3.xxy + p3.yxx)*p3.zyx);
}

float box(vec2 p, vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.)) + min(max(d.x, d.y), 0.);
}

#define col(x) vec4(lch2rgb(vec3(.8, .1, .5 * (x) + time * .06 - glow - bass)), 0)

void mainImage(out vec4 O, vec2 I)
{
    O = vec4(0);
    
    float time = iTime - 2.25,
          fade = smoothstep(-2.25, 0., time),
          bass = fract(time / 20.),
          glow = exp(-20. * bass);
    
    vec2 p   = (1.2 + fade * .1) * (I + I - iResolution.xy) / iResolution.y,
         pos = p; pos.x += 1.22;

    for(float i = .8; i < 1.01; i += .1 / 7.)
    {
        float T = 4. / i,
              h = .4 / i / i,
              t = mod(time, T),
              d = 6. * box(pos - vec2(0, .2 * t * (T - t) - h), vec2(.06, .95 - h)),
              g = 1. / max(d, .01);

        if(abs(pos.x) < .07 && pos.y > -1.) g += .25 * exp(-6. * p.y);
        
        g *= min(t, .1) * exp(4. - 20. * t / T);

        O += col(5. * i - 4.) * (g + .1 / (abs(d) + .002));
        pos.x -= .1743;
    }
    
    float con = box(p, vec2(1.33, 1));
    
    for(int i = 0; con > 0. && i < 9; i++)
    {
        float j = float(i),
              s = exp(mod(j, 3.)), g;
        
        pos = 4. * (p - vec2(sin(j *= .7), cos(j)) * (.15 * time - glow - bass)) / s;
        
        vec3 h = hash(floor(pos).xyx + j);
        
        g = sin(h.x + time + j);
        
        if(h.x < s * .05 && g > 0.) O += 2. * col(j / 9.) * g * (s * max(.5 - length(fract(pos) - .5), 0.) + (abs(box(fract(pos + .6 * h.yz - .3) - .5, vec2(.2))) < .01 ? 5. : 0.));
    }
    
    glow += .2;
    
    O = sqrt(1. - exp(-(O + col(p.y * .5 + .5) * (1. / (abs(con) + .001) + max(.8 * bass - .6, 0.) / (abs(box(p, vec2(.665, .5) * (3. - bass))) + .001))) * fade * min(bass * 1e2, 1.) * glow / (1. + 3. * dot(p, p))));

    }

"""

sound_code = """
vec2 f(float a, float b, float t)
{
    t = a * mod(t, b);
    return vec2(cos(t / 1e2), sin(t / 1e2)) * (cos(4. * t) + mix(sin(2. * t), sin(8. * t), .2 * t / a / b)) / 2. * min(.1 * t, 1.) * exp(-t / b / 2e2);
}

vec2 mainSound(int samp, float t)
{
    t -= 2.25;
    
    vec2 tot = vec2(0);
    
    vec4 C;
    
    for(int i = 0; i < 15; i++)
    {
        float T = 4. / (1. - float(i) / 70.);
        
        switch(int(floor(t / T) * T / 20.) % 4)
        {
            case 0:
            case 2: C = vec4(2160, 2025, 1620, 1350); break;
            case 1: C = vec4(1800, 1620, 1350, 1080); break;
            case 3: C = vec4(1440, 1350, 1080, 900); break;
        }
        
        tot += .5 * f(C[i % 4] / exp2(float(i / 4)), T, t);
    }
    
    return smoothstep(-2.25, 0., t) * (tot + f(C.x / 16., 20., t) + f(C.x / 8., 20., t)) * .1;
}
"""

from shadertoy import Shadertoy

if __name__ == "__main__":
    shader = Shadertoy(main_code, sound_code=sound_code)
    shader.show()
