from shadertoy import Shadertoy, DataChannel

# https://www.shadertoy.com/view/ldXXDj

main_code = """
// Created by inigo quilez - iq/2014
//   https://www.youtube.com/c/InigoQuilez
//   https://iquilezles.org/
// Creative Commons license.


// A simple and cheap 2D shader to accompany the Pirates of the Caribean music.


float fbm( vec2 p )
{
    return 0.5000*texture( iChannel1, p*1.00 ).x + 
           0.2500*texture( iChannel1, p*2.02 ).x + 
           0.1250*texture( iChannel1, p*4.03 ).x + 
           0.0625*texture( iChannel1, p*8.04 ).x;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float time = mod( iTime, 60.0 );
	vec2 p = (2.0*fragCoord-iResolution.xy) / iResolution.y;
    vec2 i = p;

    // camera
    p += vec2(1.0,3.0)*0.001*2.0*cos( iTime*5.0 + vec2(0.0,1.5) );    
    p += vec2(1.0,3.0)*0.001*1.0*cos( iTime*9.0 + vec2(1.0,4.5) );    
    float an = 0.3*sin( 0.1*time );
    float co = cos(an);
    float si = sin(an);
    p = mat2( co, -si, si, co )*p*0.85;
    
    // water
    vec2 q = vec2(p.x,1.0)/p.y;
    q.y -= 0.9*time;    
    vec2 off = texture( iChannel0, 0.1*q*vec2(1.0,2.0) - vec2(0.0,0.007*iTime) ).xy;
    q += 0.4*(-1.0 + 2.0*off);
    vec3 col = 0.2*sqrt(texture( iChannel0, 0.05*q *vec2(1.0,4.0) + vec2(0.0,0.01*iTime) ).zyx);
    float re = 1.0-smoothstep( 0.0, 0.7, abs(p.x-0.6) - abs(p.y)*0.5+0.2 );
    col += 1.0*vec3(1.0,0.9,0.73)*re*0.2*(0.1+0.9*off.y)*5.0*(1.0-col.x);
    float re2 = 1.0-smoothstep( 0.0, 2.0, abs(p.x-0.6) - abs(p.y)*0.85 );
    col += 0.7*re2*smoothstep(0.35,1.0,texture( iChannel1, 0.075*q *vec2(1.0,4.0) ).x);
    
    // sky
    vec3 sky = vec3(0.0,0.05,0.1)*1.4;
    // stars    
    sky += 0.5*smoothstep( 0.95,1.00,texture( iChannel1, p ).x);
    sky += 0.5*smoothstep( 0.85,1.0,texture( iChannel1, p ).x);
    sky += 0.2*pow(1.0-max(0.0,p.y),2.0);
    // clouds    
    float f = fbm( 0.002*vec2(p.x,1.0)/p.y );
    vec3 cloud = vec3(0.3,0.4,0.5)*0.7*(1.0-0.85*smoothstep(0.4,1.0,f));
    sky = mix( sky, cloud, 0.95*smoothstep( 0.4, 0.6, f ) );
    sky = mix( sky, vec3(0.33,0.34,0.35), pow(1.0-max(0.0,p.y),2.0) );
    col = mix( col, sky, smoothstep(0.0,0.1,p.y) );

    // horizon
    col += 0.1*pow(clamp(1.0-abs(p.y),0.0,1.0),9.0);

    // moon
    float d = length(p-vec2(0.6,0.5));
    vec3 moon = vec3(0.98,0.97,0.95)*(1.0-0.1*smoothstep(0.2,0.5,f));
    col += 0.8*moon*exp(-4.0*d)*vec3(1.1,1.0,0.8);
    col += 0.2*moon*exp(-2.0*d);
    moon *= 0.85+0.15*smoothstep(0.25,0.7,fbm(0.05*p+0.3));
    col = mix( col, moon, 1.0-smoothstep(0.2,0.22,d) );
    
    // postprocess
    col = pow( 1.4*col, vec3(1.5,1.2,1.0) );    
    col *= clamp(1.0-0.3*length(i), 0.0, 1.0 );

    // fade
    col *=       smoothstep( 3.0, 6.0,time);
    col *= 1.0 - smoothstep(44.0,50.0,time);

    fragColor = vec4( col, 1.0 );
}
"""

sound_code = """
// Created by inigo quilez - iq/2014
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

//----------------------------------------------------------------------------------------
// main instrument
float instrument( float freq, float time )
{
    float ph = 1.0;
    ph *= sin(6.283185*freq*time*2.0);
    ph *= 0.5+0.5*max(0.0,5.0-0.01*freq);
    ph *= exp(-time*freq*0.2);
    
    float y = 0.0;
    y += 0.70*sin(1.00*6.283185*freq*time+ph)*exp2(-0.7*0.007*freq*time);
    y += 0.20*sin(2.01*6.283185*freq*time+ph)*exp2(-0.7*0.011*freq*time);
    y += 0.20*sin(3.01*6.283185*freq*time+ph)*exp2(-0.7*0.015*freq*time);
    y += 0.16*sin(4.01*6.283185*freq*time+ph)*exp2(-0.7*0.018*freq*time);
    y += 0.13*sin(5.01*6.283185*freq*time+ph)*exp2(-0.7*0.021*freq*time);
    y += 0.10*sin(6.01*6.283185*freq*time+ph)*exp2(-0.7*0.027*freq*time);
    y += 0.09*sin(8.01*6.283185*freq*time+ph)*exp2(-0.7*0.030*freq*time);
    y += 0.07*sin(9.01*6.283185*freq*time+ph)*exp2(-0.7*0.033*freq*time);

    y += 0.35*y*y*y;
    y += 0.10*y*y*y;
       
    y *= 1.0 + 1.5*exp(-8.0*time);
    y *= clamp( time/0.004, 0.0, 1.0 );

    y *= 2.5-1.5*clamp( log2(freq)/10.0,0.0,1.0);
	return y;	
}


// music data
float doChannel1( float soundTime );
float doChannel2( float soundTime );

//----------------------------------------------------------------------------------------
// sound shader entrypoint
//
// input: time in seconds
// ouput: stereo wave valuie at "time"
//----------------------------------------------------------------------------------------

vec2 mainSound( in int samp, float time )
{	
    time = mod( time, 60.0 );
    
    vec2 y = vec2(0.0);
    y += vec2(0.7,0.3)*doChannel1( time ); // main instrument
    y += vec2(0.3,0.7)*doChannel2( time ); // secondary instrument
	y *= 0.1;
    
	return y;
}

//----------------------------------------------------------------------------------------

#define D(a) b+=float(a);if(t>b)x=b;

//----------------------------------------------------------------------------------------

#define tint 0.144

float doChannel1( float t )
{
  float x = 0.0;
  float y = 0.0;
  float b = 0.0;
  t /= tint;

  // F2
  x = t; b = 0.0;
  D(36)D(2)D(2)D(20)D(2)D(16)D(6)D(2)D(226)
  y += instrument( 174.0, tint*(t-x) );

  // G2
  x = t; b = 0.0;
  D(53)D(208)
  y += instrument( 195.0, tint*(t-x) );

  // A2
  x = t; b = 0.0;
  D(34)D(2)D(2)D(2)D(1)D(7)D(2)D(2)D(2)D(1)D(3)D(8)D(2)D(8)D(2)D(4)D(2)D(2)D(2)D(1)
  D(31)D(2)D(4)D(138)D(46)D(2)
  y += instrument( 220.0, tint*(t-x) );

  // A#2
  x = t; b = 0.0;
  D(42)D(2)D(2)D(14)D(2)D(2)D(1)D(25)D(2)D(16)D(2)D(2)
  y += instrument( 233.0, tint*(t-x) );

  // B2
  x = t; b = 0.0;
  D(125)
  y += instrument( 246.0, tint*(t-x) );

  // C3
  x = t; b = 0.0;
  D(35)D(6)D(7)D(2)D(3)D(1)D(5)D(7)D(2)D(2)D(1)D(1)D(2)D(3)D(6)D(199)D(2)D(2)D(2)D(1)
  y += instrument( 261.0, tint*(t-x) );

  // C#3
  x = t; b = 0.0;
  D(120)D(2)D(4)D(132)D(1)D(5)D(42)D(2)
  y += instrument( 277.0, tint*(t-x) );

  // D3
  x = t; b = 0.0;
  D(0)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(1)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(1)D(2)
  D(1)D(2)D(1)D(2)D(1)D(3)D(2)D(2)D(2)D(2)D(2)D(1)D(5)D(3)D(5)D(2)D(2)D(12)D(2)D(6)
  D(2)D(2)D(2)D(2)D(2)D(1)D(1)D(2)D(5)D(3)D(2)D(2)D(2)D(3)D(3)D(6)D(1)D(136)D(9)D(2)
  D(2)D(2)D(1)D(17)D(2)D(2)D(2)D(1)D(11)
  y += instrument( 293.0, tint*(t-x) );

  // E3
  x = t; b = 0.0;
  D(41)D(7)D(2)D(15)D(7)D(2)D(27)D(6)D(13)D(2)D(4)D(132)D(1)D(23)D(2)D(2)D(2)D(18)D(4)
  y += instrument( 329.0, tint*(t-x) );

  // F3
  x = t; b = 0.0;
  D(42)D(2)D(2)D(20)D(2)D(2)D(19)D(11)D(2)D(6)D(2)D(4)D(5)D(5)D(8)D(2)D(2)D(20)D(2)D(16)
  D(6)D(2)D(82)D(4)D(2)D(2)D(2)D(2)D(1)D(12)D(5)D(2)D(2)D(2)D(1)D(7)
  y += instrument( 349.0, tint*(t-x) );

  // G3
  x = t; b = 0.0;
  D(47)D(24)D(19)D(2)D(2)D(2)D(2)D(3)D(11)D(37)D(120)D(13)D(2)D(2)D(2)D(18)
  y += instrument( 391.0, tint*(t-x) );

  // A3
  x = t; b = 0.0;
  D(95)D(5)D(2)D(12)D(16)D(2)D(2)D(2)D(1)D(7)D(2)D(2)D(2)D(1)D(3)D(8)D(2)D(8)D(2)D(4)
  D(2)D(2)D(2)D(1)D(31)D(2)D(4)D(2)D(2)D(12)D(1)D(1)D(30)D(2)D(2)D(3)D(12)D(5)D(2)D(2)
  D(3)
  y += instrument( 440.0, tint*(t-x) );

  // A#3
  x = t; b = 0.0;
  D(96)D(2)D(40)D(2)D(2)D(14)D(2)D(2)D(1)D(25)D(2)D(16)D(2)D(2)D(24)D(18)D(1)D(1)D(24)D(24)
  y += instrument( 466.0, tint*(t-x) );

  // C4
  x = t; b = 0.0;
  D(131)D(6)D(7)D(2)D(3)D(1)D(5)D(7)D(2)D(2)D(1)D(1)D(2)D(3)D(6)D(47)D(2)
  y += instrument( 523.0, tint*(t-x) );

  // C#4
  x = t; b = 0.0;
  D(216)D(2)D(3)
  y += instrument( 554.0, tint*(t-x) );

  // D4
  x = t; b = 0.0;
  D(132)D(2)D(2)D(2)D(2)D(2)D(1)D(5)D(3)D(5)D(2)D(2)D(12)D(2)D(6)D(2)D(2)D(2)D(2)D(2)
  D(1)D(1)D(2)D(5)D(3)D(2)D(2)D(2)D(3)D(3)D(6)D(2)D(2)D(4)D(4)D(2)D(5)D(7)D(5)
  y += instrument( 587.0, tint*(t-x) );

  // E4
  x = t; b = 0.0;
  D(137)D(7)D(2)D(15)D(7)D(2)D(27)D(6)D(13)D(2)D(8)
  y += instrument( 659.0, tint*(t-x) );

  // F4
  x = t; b = 0.0;
  D(138)D(2)D(2)D(20)D(2)D(2)D(19)D(11)D(2)D(6)D(2)D(4)D(5)D(13)D(2)D(1)D(4)D(3)
  y += instrument( 698.0, tint*(t-x) );

  // G4
  x = t; b = 0.0;
  D(143)D(24)D(19)D(2)D(2)D(2)D(2)D(3)D(11)D(24)D(14)D(4)
  y += instrument( 783.0, tint*(t-x) );

  // A4
  x = t; b = 0.0;
  D(191)D(5)D(2)D(12)D(24)
  y += instrument( 880.0, tint*(t-x) );

  // A#4
  x = t; b = 0.0;
  D(192)D(2)D(52)
  y += instrument( 932.0, tint*(t-x) );

  // C5
  x = t; b = 0.0;
  y += instrument( 1046.0, tint*(t-x) );
  return y;
}

float doChannel2( float t )
{
  float x = 0.0;
  float y = 0.0;
  float b = 0.0;
  t /= tint;

  // D0
  x = t; b = 0.0;
  D(24)D(6)D(3)
  y += instrument( 36.0, tint*(t-x) );

  // F0
  x = t; b = 0.0;
  D(66)D(2)D(1)D(2)D(91)D(2)D(1)D(2)
  y += instrument( 43.0, tint*(t-x) );

  // G0
  x = t; b = 0.0;
  D(96)D(2)D(1)D(2)D(91)D(2)D(1)D(2)D(49)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  y += instrument( 48.0, tint*(t-x) );

  // A0
  x = t; b = 0.0;
  D(48)D(2)D(1)D(2)D(22)D(2)D(43)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(22)D(2)
  D(43)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  D(37)D(2)D(1)D(2)
  y += instrument( 55.0, tint*(t-x) );

  // A#0
  x = t; b = 0.0;
  D(42)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)
  D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(23)
  y += instrument( 58.0, tint*(t-x) );

  // C1
  x = t; b = 0.0;
  D(41)D(31)D(2)D(63)D(31)D(2)D(56)D(2)D(2)D(52)D(2)D(1)D(2)
  y += instrument( 65.0, tint*(t-x) );

  // D1
  x = t; b = 0.0;
  D(24)D(6)D(3)D(3)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)
  D(1)D(2)D(7)D(2)D(1)D(2)D(13)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)
  D(2)D(13)D(2)D(1)D(2)D(7)D(2)D(1)D(2)D(7)D(2)D(46)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)
  D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(7)
  y += instrument( 73.0, tint*(t-x) );

  // F1
  x = t; b = 0.0;
  D(66)D(2)D(1)D(2)D(91)D(2)D(1)D(2)D(121)D(2)D(1)D(1)D(1)
  y += instrument( 87.0, tint*(t-x) );

  // G1
  x = t; b = 0.0;
  D(96)D(2)D(1)D(2)D(91)D(2)D(1)D(2)D(49)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  y += instrument( 97.0, tint*(t-x) );

  // A1
  x = t; b = 0.0;
  D(48)D(2)D(1)D(2)D(22)D(2)D(43)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(22)D(2)
  D(43)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(1)D(2)D(1)D(2)
  D(37)D(2)D(1)D(2)
  y += instrument( 110.0, tint*(t-x) );

  // A#1
  x = t; b = 0.0;
  D(42)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)
  D(13)D(2)D(1)D(2)D(25)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(23)
  y += instrument( 116.0, tint*(t-x) );

  // C2
  x = t; b = 0.0;
  D(41)D(31)D(2)D(63)D(31)D(2)D(56)D(2)D(2)D(52)D(2)D(1)D(2)
  y += instrument( 130.0, tint*(t-x) );

  // D2
  x = t; b = 0.0;
  D(36)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)D(1)D(2)D(7)
  D(2)D(1)D(2)D(13)D(2)D(1)D(15)D(2)D(1)D(2)D(19)D(2)D(1)D(2)D(1)D(2)D(1)D(2)D(13)D(2)
  D(1)D(2)D(7)D(2)D(1)D(2)D(7)D(2)D(46)D(2)D(1)D(2)D(1)D(2)D(1)D(1)D(1)D(13)D(2)D(1)
  D(2)D(1)D(2)D(1)D(1)D(1)D(7)
  y += instrument( 146.0, tint*(t-x) );

  // F2
  x = t; b = 0.0;
  D(288)D(2)D(1)D(1)D(1)
  y += instrument( 174.0, tint*(t-x) );
  return y;
}
"""

import imageio.v3 as iio
from pathlib import Path

if __name__ == "__main__":
    shader = Shadertoy(main_code, sound_code=sound_code)

    stars_img = iio.imread(Path(__file__).parent / "media"/"stars.jpg")
    noise_img = iio.imread(Path(__file__).parent / "media"/"gray_noise_medium.png")

    shader.main_pass.channel_0 = DataChannel(stars_img, filter="mipmap")
    shader.main_pass.channel_1 = DataChannel(noise_img, filter="mipmap")

    shader.show()
