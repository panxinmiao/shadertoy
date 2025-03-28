# https://www.shadertoy.com/view/XlfGRj

shader_code = """
// migrated from: https://www.shadertoy.com/view/XlfGRj, By Kali
const iterations = 17;
const formuparam = 0.53;
const volsteps = 20;
const stepsize = 0.1;
const zoom = 0.800;
const tile = 0.850;
const speed = 0.010;
const brightness = 0.0015;
const darkmatter = 0.300;
const distfading = 0.730;
const saturation = 0.850;
fn _mod( p1 : vec3<f32>, p2 : vec3<f32> ) -> vec3<f32> {
    let mx = p1.x - p2.x * floor( p1.x / p2.x );
    let my = p1.y - p2.y * floor( p1.y / p2.y );
    let mz = p1.z - p2.z * floor( p1.z / p2.z );
    return vec3<f32>(mx, my, mz);
}
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    var uv = frag_coord.xy / i_resolution.xy - vec2<f32>(0.5, 0.5);
    uv.y *= i_resolution.y / i_resolution.x;
    var dir = vec3<f32>(uv * zoom, 1.0);
    let time = i_time * speed + 0.25;
    //mouse rotation
    let a1 = 0.5 + i_mouse.x / i_resolution.x * 2.0;
    let a2 = 0.8 + i_mouse.y / i_resolution.y * 2.0;
    let rot1 = mat2x2<f32>(cos(a1), sin(a1), -sin(a1), cos(a1));
    let rot2 = mat2x2<f32>(cos(a2), sin(a2), -sin(a2), cos(a2));
    let dir_xz = dir.xz * rot1;
    dir.x = dir_xz.x;
    dir.z = dir_xz.y;
    let dir_xy = dir.xy * rot2;
    dir.x = dir_xy.x;
    dir.y = dir_xy.y;
    var _from = vec3<f32>(1.0, 0.5, 0.5);
    _from += vec3<f32>(time * 2.0, time, -2.0);
    let from_xz = _from.xz * rot1;
    _from.x = from_xz.x;
    _from.z = from_xz.y;
    let from_xy = _from.xy * rot2;
    _from.x = from_xy.x;
    _from.y = from_xy.y;
    //volumetric rendering
    var s = 0.1;
    var fade = 1.0;
    var v = vec3<f32>(0.0);
    for (var r: i32 = 0; r < volsteps; r = r + 1) {
        var p = _from + s * dir * 0.5;
        p = abs(vec3<f32>(tile) - _mod(p, vec3<f32>(tile * 2.0))); // tiling fold
        var pa = 0.0;
        var a = 0.0;
        for (var i : i32 = 0; i < iterations; i = i + 1) {
            p = abs(p) / dot(p, p) - formuparam; // the magic formula
            a += abs(length(p) - pa); // absolute sum of average change
            pa = length(p);
        }
        let dm = max(0.0, darkmatter - a * a * 0.001); //dark matter
        a = a * a * a; // add contrast
        if (r > 6) {
            fade = fade * (1.0 - dm); // dark matter, don't render near
        }
        v += vec3<f32>(fade);
        v += vec3<f32>(s, s * s, s * s * s * s) * a * brightness * fade; // coloring based on distance
        fade = fade * distfading; // distance fading
        s = s + stepsize;
    }
    v = mix(vec3<f32>(length(v)), v, saturation); //color adjust
    return vec4<f32>(v * 0.01, 1.0);
}
"""

from shadertoy import Shadertoy

if __name__ == "__main__":
    shader = Shadertoy(shader_code)
    shader.show()
