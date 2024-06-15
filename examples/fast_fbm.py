# https://www.shadertoy.com/view/msXSR2

main_code = """
/* A new attempt at fast fbm terrain raymarching  
   combining multiple techniques I've learned so far:
   
   - Break the fbm loop early if the point is located above the current noise
     value to prevent unecessary noise calculations. (BREAK_EARLY) 
     
   - Increase the minimum distance considered as a surface hit the further
     away from the camera the point is. (RELAXATION)
     
   - Calculate only a smaller amount of fbm layers when raymarching
     shadows. (SHADOW_FBM_LAYERS)
     
   - I also tried to use a simpler and faster noise function as seen in
     this shader: https://www.shadertoy.com/view/NscGWl. 
     However, the terrain gets sharper so I decided to use a mix of this 
     new noise and value noise: for the first 4 fbm layers (the ones that 
     are computed most frequently), I use this fast noise but for the 
     later layers I use value noise. This is a good compromise
     between performance and nice terrain shape. (FAST_LAYERS)
     
   - Finally I added a visual improvement thanks to @Dave_Hoskins's comment,
     the step size for the normal calculation increases with the distance in
     order to reduce aliasing artefacts (REDUCE_ALIASING)
     
   Left panel: total number of fbm loops executed during the whole raymarching
   (blue: low values, red: high values). Slide with the mouse.
*/

// Set to 0 to disable all optimizations (except FAST_LAYERS that changes the terrain)
// On my machine the improvement is pretty huge, 40fps vs 20fps.

#define OPTIMIZE 1       // <-- Change this to compare the performances

// Total number of fbm layers
// Increase this to make your fps drop

#define FBM_LAYERS 10

// Number of layers that will use the fast noise function
// instead of value noise (in addition to the first layer)

#define FAST_LAYERS 3

// Visual improvement only: reduce aliasing in the distance

#define REDUCE_ALIASING 1

#if OPTIMIZE
    // Break out of the fbm loop when possible
    #define BREAK_EARLY       1
    // Number of layers computed while raymarching shadows
    #define SHADOW_FBM_LAYERS 5
    // Min. Surface distance increase
    #define RELAXATION        0.2
#else
    // Default parameters (no optimization)
    #define BREAK_EARLY       0
    #define SHADOW_FBM_LAYERS FBM_LAYERS
    #define RELAXATION        0.
#endif

float total_fbm_loop_count = 0.; 

float fastnoise2(vec2 p) {
    return (sin(p.x)-cos(p.y))*.5+.5;
}

float fbm(vec2 p, float h, int layers) {
    float n = fastnoise2(p); // The first layer is calculated with fast noise
    float a = 1.;
    
    for (int i = 0; i < layers; i++) {
#if BREAK_EARLY  
        if (h > n + a*2.) break; // Break early if the point is above the noise
#endif 
        
        p *= 2.; a *= .5;
        total_fbm_loop_count++;
   
        if (i < FAST_LAYERS) // Fast calculations for the first layers
            n -= a*abs(fastnoise2(p)-n);
        else                 // Value noise for the last layers
            n -= a*noise2(p);      
    } 
    
    return n;
}

float map(vec3 p, int layers) {
    p.y *= .8;
    p.y -= .6;
    float terrain = p.y - fbm(p.xz*.5, p.y, layers);
    
    return terrain;
}

#define MAX_ITERATIONS 100.
#define MAX_DISTANCE 100.
#define EPSILON 0.001

vec3 getNormal(vec3 p, float d) {
    vec2 e = vec2(1.0,-1.0)*0.5773*0.0025;
    
#if REDUCE_ALIASING
    e *= d; // Increase the step size with the distance to reduce aliasing
#endif

    return normalize(e.xyy*map(p + e.xyy, FBM_LAYERS) + e.yyx*map(p + e.yyx, FBM_LAYERS) + 
					 e.yxy*map(p + e.yxy, FBM_LAYERS) + e.xxx*map(p + e.xxx, FBM_LAYERS));
}

float getShadow(vec3 ro, vec3 rd, float maxt) {
    float t = 0.1, res = 1., k = 10.;
    for (float i = 0.; i < MAX_ITERATIONS; i++) {
        vec3 p = ro + t*rd;
        // Use a smaller number of fbm terrain layers for the shadow
        float d = map(p, SHADOW_FBM_LAYERS);
        res = min(res, k*d/t);
        t += d*1.;
        if (t > maxt) return res;
        if (d < EPSILON) return 0.;
    }
    return 0.;
}

#define rot(a) mat2(cos(a), -sin(a), sin(a), cos(a))
void initRayOriginAndDirection(vec2 uv, inout vec3 ro, inout vec3 rd) {
    vec2 m = iMouse.z == 0. ? vec2(.5) : iMouse.xy/iResolution.xy*2.-1.; 
    ro = vec3(0., 3., 3.);
    ro.zx *= rot(iTime*.1); 
    vec3 f = normalize(vec3(0.,1.,0.)-ro), r = normalize(cross(vec3(0,1,0), f));
    rd = normalize(f + uv.x*r + uv.y*cross(f, r));
}

void mainImage(out vec4 O, in vec2 F) {
    vec2 uv = (2.*F - iResolution.xy)/iResolution.y;
    vec3 p, ro, rd, col = vec3(1.);
    float t = 0., d;

    initRayOriginAndDirection(uv, ro, rd);
        
    for (float i = 0.; i < MAX_ITERATIONS; i++) {
        p = ro + t*rd;
        d = map(p, FBM_LAYERS);
        t += d > 0. ? d*.9 : d*.2;
        // EPSILON is the minimum surface distance considered as a hit.
        // We increase it the further away as we don't need as much details
        if (abs(d) < EPSILON*(1. + t*RELAXATION) || t > MAX_DISTANCE) break;
    }
    
    float phi = 0.32 * 6.28, the = -0.00 * 3.14 + 1.27;
    vec3 lightDir = normalize(vec3(sin(the)*sin(phi), cos(the), sin(the)*cos(phi)));
    vec3 skyColor = vec3(1.2,.8,.3);
    
    if (t < MAX_DISTANCE) {
        vec3 n = getNormal(p - rd*EPSILON*4., t);
        float sunLight    = max(.0, dot(n, lightDir));
        float sunShadow   = max(.0, getShadow(p + n*EPSILON*4., lightDir, MAX_DISTANCE));
        float skyLight    = max(.0, n.y);
        float bounceLight = max(.0, dot(n, -lightDir));
        float spec        = max(.0, dot((rd + n)/2., lightDir));
        
        col *= 1.0*vec3(1.,1.,.8) * sunLight * sunShadow;
        col += 0.1*skyColor * skyLight;
        col += 0.1*vec3(.4,.2,0.) * bounceLight;
        col += 0.8*vec3(1.)*pow(spec, 4.);
    }
         
    float groundFog = min(.5-ro.y/rd.y, t) - (0.1-ro.y)/rd.y;
    if (groundFog > 0.) {
        col = mix(vec3(1.,.8,.6), col, exp(-groundFog));
    }
    
    vec3 fog = exp2(-t*0.05*vec3(1,1.8,4)); 
    col = mix(clamp(skyColor - vec3(1.,1.5,2.)*abs(rd.y)*.7, vec3(0.), vec3(1.)), col, fog);   
    
    col = pow(col, vec3(.4545));
    
    float mp = iMouse.z == 0. ? -smoothstep(0., 3., iTime)*.6+cos(iTime*3.)*.04 : 3.55*(iMouse.x/iResolution.x-.5);
    if (uv.x < mp) col = palette(0.014 * total_fbm_loop_count / float(FBM_LAYERS));
       
    O = vec4(col, 1.0);
}
"""

common_code = """
// Value noise - https://www.shadertoy.com/view/lsf3WH
float hash(vec2 p) {
    p  = 50.0*fract( p*0.3183099 + vec2(0.71,0.113));
    return fract( p.x*p.y*(p.x+p.y) );
}
float noise2( in vec2 p ) {
    vec2 i = floor( p );
    vec2 f = fract( p );
	vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( hash( i + vec2(0.0,0.0) ), 
                     hash( i + vec2(1.0,0.0) ), u.x),
                mix( hash( i + vec2(0.0,1.0) ), 
                     hash( i + vec2(1.0,1.0) ), u.x), u.y);
}

// Heatmap color palette - https://www.shadertoy.com/view/wlGcWG
vec3 palette( float h ) {
    vec3 col =    vec3(0.0,0.3,1.0);
    col = mix(col,vec3(1.0,0.8,0.0),smoothstep(0.33-0.2,0.33+0.2,h));
    col = mix(col,vec3(1.0,0.0,0.0),smoothstep(0.66-0.2,0.66+0.2,h));
    col.y += 0.5*(1.0-smoothstep(0.0,0.2,abs(h-0.33)));
    col *= 0.5 + 0.5*h;
    return col;
}
"""

from shadertoy import Shadertoy

if __name__ == "__main__":
    shader = Shadertoy(main_code, common_code=common_code)
    shader.show()
