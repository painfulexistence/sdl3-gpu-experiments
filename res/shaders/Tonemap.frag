#version 450

layout(location = 0) in vec2 tex_uv;

layout(location = 0) out vec4 Color;

layout(set = 2, binding = 0) uniform sampler2D hdrTexture;

layout(set = 3, binding = 0) uniform TonemapParams {
    float exposure;
};


const float gamma = 2.2;
const float invGamma = 1.0 / gamma;

vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 color = texture(hdrTexture, tex_uv).rgb;

    color *= exposure;

    color = aces(color);

    color = pow(color, vec3(invGamma));

    Color = vec4(color, 1.0);
}