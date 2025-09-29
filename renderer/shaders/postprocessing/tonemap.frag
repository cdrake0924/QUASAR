#include "tonemap.glsl"

out vec4 FragColor;

in vec2 TexCoord;

uniform sampler2D screenColor;
uniform sampler2D screenDepth;
uniform sampler2D screenNormals;
uniform sampler2D screenPositions;
uniform usampler2D idTexture;

uniform bool tonemap;
uniform float exposure;

void main() {
    vec3 color = texture(screenColor, TexCoord).rgb;
    if (tonemap) {
        color = tonemapFilmic(color, exposure);
    }
    FragColor = vec4(color, 1.0);
}
