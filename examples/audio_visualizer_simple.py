# https://www.shadertoy.com/view/NlVGDK

main_code = """
#define SPECTRUM_BARS 30

// colormap https://www.shadertoy.com/view/WlfXRN, https://www.shadertoy.com/view/3lBXR3

vec3 inferno(float t)
{
	const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
	const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
	const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
	const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
	const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
	const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
	const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);

	return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}


vec3 turbo(float t) {
	const vec3 c0 = vec3(0.1140890109226559, 0.06288340699912215, 0.2248337216805064);
	const vec3 c1 = vec3(6.716419496985708, 3.182286745507602, 7.571581586103393);
	const vec3 c2 = vec3(-66.09402360453038, -4.9279827041226, -10.09439367561635);
	const vec3 c3 = vec3(228.7660791526501, 25.04986699771073, -91.54105330182436);
	const vec3 c4 = vec3(-334.8351565777451, -69.31749712757485, 288.5858850615712);
	const vec3 c5 = vec3(218.7637218434795, 67.52150567819112, -305.2045772184957);
	const vec3 c6 = vec3(-52.88903478218835, -21.54527364654712, 110.5174647748972);

	return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}


void showWave(out vec4 fragColor, in vec2 uv, in vec2 resolution)
{
	// Value
	float y = texelFetch(iChannel0, ivec2(int(uv.x * 512.0), 1), 0).x;
	// Derivation of value
	float dy = dFdx(y);
	// Average between two samples
	float center = y + dy / 2.0;
	// Výpočet veľkosti pixelu
	float pixelSize = 1.0 / resolution.y;
	// Vertial width of line
	float lineWidth = max(abs(dy), pixelSize);
	// White for zero distance
	float color = (lineWidth - abs(center - uv.y)) / lineWidth;
	// Remove negative values
	color = max(color, 0.0);
	// Final color
	fragColor = vec4(vec3(color), 1.0);
}

void showSpectrum(out vec4 fragColor, in vec2 uv, in vec2 resolution)
{
	// Bar nuber
	int barNumber = int(uv.x * float(SPECTRUM_BARS));
	// Spectrum frequency (range [0, 1])
	float frequency = (float(barNumber)/float(SPECTRUM_BARS + 1)) + 1.0 / float(SPECTRUM_BARS);
	// Load frequency
	float val = texelFetch(iChannel0, ivec2(int(frequency * 512.0), 0), 0).x;
	// Color from palette
	vec3 color = turbo(min(val * 1.1, 1.0));
	// Display bar with selected color
	if (val < uv.y || fract(uv.x * float(SPECTRUM_BARS)) < 0.2) {
		fragColor = vec4(0, 0, 0, 1);
	}
	else {
		fragColor = vec4(color, 1);
	}
}

void histogramSpectrum(out vec4 fragColor, in vec2 uv, in vec2 resolution)
{

	// Signal, x, y derivate
	vec3 signal = texture(iChannel1, uv).xyz;
	// Normal from x, y derivate
	vec3 normal = normalize(vec3(signal.y, signal.z, 0.1));
	// Bmmp mapping
	float bumpMultiplier = dot(normal, vec3(0.0, 1.0, 1.0)) * 0.5 + 0.5;
	// Final color
	fragColor = vec4(inferno(signal.x) * bumpMultiplier, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	// Koordináty [0, 1]
	vec2 uv = fragCoord.xy / iResolution.xy;

	// Vykreslenie deliacej čiary
	if (uv.x > 0.7) {
		if (int(fragCoord.y) == int(iResolution.y) / 2) {
			fragColor = vec4(vec3(0.5), 1.0);
			return;
		}
	}
	if (int(fragCoord.x) == int(float(iResolution.x) * 0.7)) {
		fragColor = vec4(vec3(0.5), 1.0);
		return;
	}

	if (uv.x > 0.7) {
		uv.x = (uv.x - 0.7) / 0.3;
		if (uv.y < 0.5) {
			showSpectrum(fragColor, uv * vec2(1.0, 2.0), iResolution.xy * vec2(1.0, 0.5));
		}
		else {
			showWave(fragColor, (uv - vec2(0.0, 0.5)) * vec2(1.0, 2.0), iResolution.xy * vec2(1.0, 0.5));
		}
	}
	else {
        histogramSpectrum(fragColor, uv * vec2(1.0 / 0.7, 1.0), iResolution.xy * vec2(0.7, 1.0));

	}
}
"""

buffer_a_code = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
	vec2 uv = fragCoord.xy / iResolution.xy;
	int freq = int(uv.y*512.0);
	float val;

	if (int(fragCoord.x) == 0) {
		val = texelFetch(iChannel0, ivec2(freq, 0), 0).x;
	}
	else {
		val = texelFetch(iChannel1, ivec2(fragCoord) + ivec2(-1, 0), 0).x;
	}
	fragColor = vec4(val, dFdx(val), dFdy(val), 1);
}
"""

from shadertoy import Shadertoy
from shadertoy.audio import AudioChannel
from pathlib import Path

if __name__ == "__main__":
    shader = Shadertoy(main_code, buffer_a_code=buffer_a_code)
    audio_channel = AudioChannel(Path(__file__).parent / "media"/"8_bit_mentality.mp3")

    shader.buffer_a_pass.channel_0 = audio_channel
    shader.buffer_a_pass.channel_1 = shader.buffer_a_pass

    shader.main_pass.channel_0 = audio_channel
    shader.main_pass.channel_1 = shader.buffer_a_pass

    audio_channel.play()
    shader.show()
