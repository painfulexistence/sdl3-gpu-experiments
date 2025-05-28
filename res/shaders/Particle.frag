#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 tex_uv;

layout(location = 0) out vec4 color;

// layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(tex_uv - center);

    float radius = 0.5;
    float softness = 0.1;
    float alpha = 1.0 - smoothstep(radius - softness, radius, dist);

    float glow = exp(-dist * 8.0);

    float ring = smoothstep(0.2, 0.22, dist) * smoothstep(0.3, 0.28, dist);

    alpha = alpha + glow * 0.3 + ring * 0.5;
    alpha = clamp(alpha, 0.0, 1.0);

    color = vec4(frag_color, alpha);
}
