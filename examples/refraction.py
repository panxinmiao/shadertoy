from shadertoy import Shadertoy

# https://www.shadertoy.com/view/flGyDd

main_code = """//glsl
// CC0: Refraction + post proc
// Added a bit of post processing to the earlier refraction experiment

#define TIME            iTime
#define RESOLUTION      iResolution
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))


// License: Unknown, author: nmz (twitter: @stormoid), found: https://www.shadertoy.com/view/NdfyRM
vec3 sRGB(vec3 t) {
  return mix(1.055*pow(t, vec3(1./2.4)) - 0.055, 12.92*t, step(t, vec3(0.0031308)));
}

// License: Unknown, author: Matt Taylor (https://github.com/64), found: https://64.github.io/tonemapping/
vec3 aces_approx(vec3 v) {
  v = max(v, 0.0);
  v *= 0.6f;
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0f, 1.0f);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec3 col = vec3(0.0);
  col = texture(iChannel0, q).xyz;
  col = aces_approx(col);
  col = sRGB(col);
  fragColor = vec4(col, 1.0);
}
"""

buffer_a_code = """//glsl
#define PI              3.141592654
#define TAU             (2.0*PI)
#define PHI             (sqrt(5.0)*0.5 + 0.5)

#define TIME            iTime
#define RESOLUTION      iResolution
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))

#define TOLERANCE       0.0001
#define MAX_RAY_LENGTH  20.0
#define MAX_RAY_MARCHES 60
#define NORM_OFF        0.001
#define MAX_BOUNCES     6

mat3 g_rot  = mat3(1.0); 
vec3 g_mat  = vec3(0.0);
vec3 g_beer = vec3(0.0);

// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
const vec4 hsv2rgb_K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www);
  return c.z * mix(hsv2rgb_K.xxx, clamp(p - hsv2rgb_K.xxx, 0.0, 1.0), c.y);
}
// License: WTFPL, author: sam hocevar, found: https://stackoverflow.com/a/17897228/418488
//  Macro version of above to enable compile-time constants
#define HSV2RGB(c)  (c.z * mix(hsv2rgb_K.xxx, clamp(abs(fract(c.xxx + hsv2rgb_K.xyz) * 6.0 - hsv2rgb_K.www) - hsv2rgb_K.xxx, 0.0, 1.0), c.y))

const float hoff = 0.0;

const vec3 skyCol     = (vec3(hoff+0.14, 0.484, 1.0));
const vec3 glowCol    = (vec3(hoff+6.0, 3.072, 1.20));
const vec3 diffuseCol = (vec3(hoff+0.15, 0.49, 1.0));
const vec3 lightPos   = vec3(0.0, 10.0, 0.0);

const float initt       = 0.1; 

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
float box(vec2 p, vec2 b) {
  vec2 d = abs(p)-b;
  return length(max(d,0.0)) + min(max(d.x,d.y),0.0);
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/www/articles/intersectors/intersectors.htm
float rayPlane(vec3 ro, vec3 rd, vec4 p) {
  return -(dot(ro,p.xyz)+p.w)/dot(rd,p.xyz);
}

mat3 rot_z(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      c,s,0
    ,-s,c,0
    , 0,0,1
    );
}

mat3 rot_y(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      c,0,s
    , 0,1,0
    ,-s,0,c
    );
}

mat3 rot_x(float a) {
  float c = cos(a);
  float s = sin(a);
  return mat3(
      1, 0,0
    , 0, c,s
    , 0,-s,c
    );
}

float sphere(vec3 p, float r) {
  return length(p) - r;
}

// License: MIT, author: Inigo Quilez, found: https://iquilezles.org/articles/distfunctions2d/
float flatTorus(vec3 p, vec2 dim) {
  float d = length(p.xy)-dim.x;
  d = abs(d) - dim.y;
  vec2 w = vec2(d, abs(p.z) - dim.y);
  return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}

vec3 skyColor(vec3 ro, vec3 rd) {
  vec3 col = clamp(vec3(0.0025/abs(rd.y))*skyCol, 0.0, 1.0);

  float tp0  = rayPlane(ro, rd, vec4(vec3(0.0, 1.0, 0.0), 4.0));
  float tp1  = rayPlane(ro, rd, vec4(vec3(0.0, -1.0, 0.0), 6.0));
  float tp = tp1;
  tp = max(tp0,tp1);


  if (tp1 > 0.0) {
    vec3 pos  = ro + tp1*rd;
    vec2 pp = pos.xz;
    float db = box(pp, vec2(6.0, 9.0))-1.0;
    
    col += vec3(4.0)*skyCol*rd.y*rd.y*smoothstep(0.25, 0.0, db);
    col += vec3(0.8)*skyCol*exp(-0.5*max(db, 0.0));
  }

  if (tp0 > 0.0) {
    vec3 pos  = ro + tp0*rd;
    vec2 pp = pos.xz;
    float ds = length(pp) - 0.5;
    
    col += vec3(0.25)*skyCol*exp(-.5*max(ds, 0.0));
  }

  return clamp(col, 0.0, 10.0);
}

float df(vec3 p) {
  p *= g_rot;
  const mat2 rot0 = ROT(0.5);
  const mat2 rot1 = ROT(1.0);
  vec3 p0 = p;
  vec3 p1 = p;
  vec3 p2 = p;

  p1.zx *= rot0;
  p2.zy *= rot1;
  const float w = 0.2;
  const float rnd = 0.025;
  float d0 = flatTorus(p0, vec2(2.0, w))-rnd;
  float d1 = flatTorus(p1, vec2(1.4, w))-rnd;
  float d2 = flatTorus(p2, vec2(0.8, w))-rnd;
  float d3 = sphere(p, 0.4);
  
  vec3 mat = vec3(0.9, 0.5, 0.8);
  const vec3 gcol = -2.5*((vec3(1.0, 0.21375, 0.075)));
//  const vec3 gcol = 3.5*(1.0-HSV2RGB(vec3(0.025, 0.9, 1.0)));
  vec3 beer = gcol;
  
  float d = d0;
  d = min(d, d1);
  d = min(d, d2);
  if (d3 < d) {
    const vec3 gcol = -10.*((vec3(1.0, 0.584, 0.35)));
    beer = gcol;
    d = d3;
  }

  g_mat = mat;
  g_beer = beer;
  return d;
}

vec3 normal(vec3 pos) {
  vec2  eps = vec2(NORM_OFF,0.0);
  vec3 nor;
  nor.x = df(pos+eps.xyy) - df(pos-eps.xyy);
  nor.y = df(pos+eps.yxy) - df(pos-eps.yxy);
  nor.z = df(pos+eps.yyx) - df(pos-eps.yyx);
  return normalize(nor);
}

float rayMarch(vec3 ro, vec3 rd, float dfactor, out int ii) {
  float t = 0.0;
  float tol = dfactor*TOLERANCE;
  ii = MAX_RAY_MARCHES;
  for (int i = 0; i < MAX_RAY_MARCHES; ++i) {
    if (t > MAX_RAY_LENGTH) {
      t = MAX_RAY_LENGTH;    
      break;
    }
    float d = dfactor*df(ro + rd*t);
    if (d < TOLERANCE) {
      ii = i;
      break;
    }
    t += d;
  }
  return t;
}

vec3 render(vec3 ro, vec3 rd) {
  vec3 agg = vec3(0.0, 0.0, 0.0);
  vec3 ragg = vec3(1.0);

  bool isInside = df(ro) < 0.0;

  for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
    float dfactor = isInside ? -1.0 : 1.0;
    float mragg = min(min(ragg.x, ragg.y), ragg.z);
    if (mragg < 0.025) break;
    int iter;
    float st = rayMarch(ro, rd, dfactor, iter);
    const float mrm = 1.0/float(MAX_RAY_MARCHES);
    float ii = float(iter)*mrm;
    vec3 mat = g_mat;
    if (st >= MAX_RAY_LENGTH) {
      agg += ragg*skyColor(ro, rd);
      break; 
    }

    vec3 sp = ro+rd*st;

    vec3 sn = dfactor*normal(sp);
    float fre = 1.0+dot(rd, sn);
//    fre = clamp(abs(fre), 0.0, 1.0);
    fre *= fre;
    fre = mix(0.1, 1.0, fre);

    vec3 ld     = normalize(lightPos - sp);

    float dif   = max(dot(ld, sn), 0.0); 
    vec3 ref    = reflect(rd, sn);
    float re    = mat.z;
    float ire   = 1.0/re;
    vec3 refr   = refract(rd, sn, !isInside ? re : ire);
    vec3 rsky   = skyColor(sp, ref);
    vec3 col = vec3(0.0);    
    col += diffuseCol*dif*dif*(1.0-mat.x);
    float edge = smoothstep(1.0, 0.9, fre);
    col += rsky*mat.y*fre*vec3(1.0)*edge;
    if (isInside) {
      ragg *= exp(-st*g_beer);
    }
    agg += ragg*col;

    if (refr == vec3(0.0)) {
      rd = ref;
    } else {
      ragg *= mat.x;
      isInside = !isInside;
      rd = refr;
    }

    // TODO: if beer is active should also computer it based on initt    
    ro = sp+initt*rd;
  }

  return agg;
}

vec3 effect(vec2 p) {
  g_rot = rot_x(-0.3*TIME)*rot_z(0.5*TIME);
  vec3 ro = 0.9*vec3(0.0, 2.0, 5.0);
  const vec3 la = vec3(0.0, 0.0, 0.0);
  const vec3 up = vec3(0.0, 1.0, 0.0);

  vec3 ww = normalize(la - ro);
  vec3 uu = normalize(cross(up, ww ));
  vec3 vv = normalize(cross(ww,uu));
  const float fov = tan(TAU/6.);
  vec3 rd = normalize(-p.x*uu + p.y*vv + fov*ww);

  vec3 col = render(ro, rd);
  
  return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1. + 2. * q;
  p.x *= RESOLUTION.x/RESOLUTION.y;
  vec3 col = vec3(0.0);
  col = effect(p);
  fragColor = vec4(col, 1.0);
}

"""

buffer_b_code = """//glsl
#define TIME            iTime
#define RESOLUTION      iResolution
#define ROT(a)          mat2(cos(a), sin(a), -sin(a), cos(a))

#define PI          3.141592654
#define TAU         (2.0*PI)

mat2 brot = ROT(2.399);
// License: Unknown, author: Dave Hoskins, found: Forgot where
vec3 dblur(vec2 q,float rad) {
  vec3 acc=vec3(0);
  const float m = 0.0025;
  vec2 pixel=vec2(m*RESOLUTION.y/RESOLUTION.x,m);
  vec2 angle=vec2(0,rad);
  rad=1.;
  const int iter = 30;
  for (int j=0; j<iter; ++j) {  
    rad += 1./rad;
    angle*=brot;
    vec4 col=texture(iChannel1,q+pixel*(rad-1.)*angle);
    acc+=clamp(col.xyz, 0.0, 10.0);
  }
  return acc*(1.0/float(iter));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
  vec2 q = fragCoord/RESOLUTION.xy;
  vec2 p = -1.0+2.0*q;
  vec2 p2 = 0.9*p;
  vec2 q2 = 0.5+0.5*p2;
  const vec2 off = 0.0125*vec2(0.0, -1.0);
  p.x *= RESOLUTION.x/RESOLUTION.y;

  float pp = smoothstep(0.0, 1.0, sin(0.25*TAU*TIME));
  float bf = mix(0.66, 0.75, pp)*smoothstep(mix(0.65, 0.85, pp), 0.0, dot(p, p));
  vec3 bcol = bf > 0.05 ? dblur(q2, mix(0.5, 1.0, pp)) : vec3(0.0);

  vec3 col = vec3(0.0);
  col = texture(iChannel0, q).xyz;
  col += bcol*bf;
  fragColor = vec4(col, 1.0);
}

"""

if __name__ == "__main__":
    shader = Shadertoy(
        main_code,
        buffer_a_code=buffer_a_code,
        buffer_b_code=buffer_b_code,
    )
    # shader.buffer_a_pass.channel_0 = shader.buffer_a_pass

    shader.buffer_b_pass.channel_0 = shader.buffer_a_pass
    shader.buffer_b_pass.channel_1 = shader.buffer_b_pass

    shader.main_pass.channel_0 = shader.buffer_b_pass

    shader.show()
