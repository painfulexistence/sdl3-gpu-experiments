#version 450

layout(location = 0) in vec2 tex_uv;
layout(location = 0) out vec4 Color;
layout(set = 2, binding = 0) uniform sampler2D tex;

void main() {
	Color = texture(tex, tex_uv);
}