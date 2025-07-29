#include "tone_map.glsl"

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idBuffer;

uniform bool toneMap;
uniform float exposure;

void main() {
    vec3 color = texture(screenColor, TexCoord).rgb;
    if (toneMap) {
        color = applyToneMapExponential(color, exposure);
        color = linearToSRGB(color);
    }
    FragColor = vec4(color, 1.0);
}
