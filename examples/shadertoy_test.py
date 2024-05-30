from shadertoy import Shadertoy

if __name__ == "__main__":
    main_code = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
    let uv = frag_coord / i_resolution.xy;

    if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }else{
        return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
    }

}
"""
    shader = Shadertoy(main_code)
    shader.show()