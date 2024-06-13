from shadertoy import Shadertoy, DataChannel

# https://www.shadertoy.com/view/MfyXzV

main_code = """
// This is an example demonstrating the FIS technique for stochastically evaluating the bilinear/biquadratic/Gaussian filters.
// https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/
// Starting from top, going clockwise:
// Full bilinear, stochastic bilinear, stochastic (bi)quadratic, stochastic Gaussian.

// Click somewhere in the shader to disable the temporal accumulation.


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    vec3 col = texture(iChannel0, uv).xyz;

    // Output to screen
    fragColor = vec4(col,1.0);
}
"""

buffer_a_code = """
float rand(float co) { return fract(sin(co*(91.3458)) * 47453.5453); }
float rand(vec2 co){ return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453); }

const float PI = 3.1415926;

vec2 boxMullerTransform(vec2 u)
{
    vec2 r;
    float mag = sqrt(-2.0 * log(u.x));
    return mag * vec2(cos(2.0 * PI * u.y), sin(2.0 * PI * u.y));
}

vec3 stochastic_gauss(in vec2 uv, in vec2 rand) {
    vec2 orig_tex_coord = uv * iResolution.xy - 0.5;
    vec2 uv_full = (round(orig_tex_coord + boxMullerTransform(rand)*0.5)+0.5) / iResolution.xy;

    return texture(iChannel0, uv_full).xyz;  
}


vec3 stochastic_bilin(in vec2 uv, in vec2 rand) {
    vec2 orig_tex_coord = uv * iResolution.xy - 0.5;
    vec2 uv_full = (round(orig_tex_coord + rand - 0.5)+0.5) / iResolution.xy;

    return texture(iChannel0, uv_full).xyz;  
}

// Inverse CDF sampling for a tent / bilinear kernel. Followed by rounding - nearest-neighbor box kernel,
// UV jittering this way produces a biquadratic B-Spline kernel.
// See the paper for the explanation: https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/
vec2 bilin_inverse_cdf_sample(vec2 x) {
    return mix(1.0 - sqrt(2.0 - 2.0 * x), -1.0 + sqrt(2.0 * x), step(x, vec2(0.5)));
}

vec3 stochastic_quadratic(in vec2 uv, in vec2 rand) {
    vec2 orig_tex_coord = uv * iResolution.xy - 0.5;
    vec2 uv_full = (round(orig_tex_coord + bilin_inverse_cdf_sample(rand))+0.5) / iResolution.xy;

    return texture(iChannel0, uv_full).xyz;  
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 orig_uv = uv;
    uv *= 0.04;
    uv += 0.1;
    // 
    vec4 random = texture(iChannel2, orig_uv+vec2(rand(float(iFrame)), rand(iTime)));
    vec3 col = orig_uv.x > orig_uv.y ? stochastic_quadratic(uv, random.xy) : stochastic_gauss(uv, random.xy);

    if (1.0-orig_uv.x < orig_uv.y) {
        col = texture(iChannel0, uv).xyz;
        if (orig_uv.x > orig_uv.y) {
            col = stochastic_bilin(uv, random.xy);
        }
    }
    if (abs(1.0-orig_uv.x - orig_uv.y) < 0.005)
        col = vec3(1,1,1);
    if (abs(orig_uv.x - orig_uv.y) < 0.005)
        col = vec3(1,1,1);    
    
    if (iTime > 0.1 && iMouse.z < fragCoord.x) {
        col = col * 0.02 + texture(iChannel1, orig_uv).xyz * 0.98;
    }
    

    // Output to screen
    fragColor = vec4(col,1.0);
}
"""
import imageio.v3 as iio
from pathlib import Path

if __name__ == "__main__":
    shader = Shadertoy(
        main_code,
        buffer_a_code=buffer_a_code,
    )
    rgba_noise_img = iio.imread(Path(__file__).parent / "media"/"rgba_noise_medium.png")
    blue_noise_img = iio.imread(Path(__file__).parent / "media"/"blue_noise.png")

    shader.buffer_a_pass.channel_0 = DataChannel(rgba_noise_img, vflip=True)
    shader.buffer_a_pass.channel_1 = shader.buffer_a_pass
    shader.buffer_a_pass.channel_2 = DataChannel(blue_noise_img, filter="nearest", vflip=True)

    shader.main_pass.channel_0 = shader.buffer_a_pass

    shader.show()
