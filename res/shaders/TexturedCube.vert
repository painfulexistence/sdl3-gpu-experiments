#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec2 tex_uv;

layout(set = 1, binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
};
layout(set = 1, binding = 1) uniform Instance {
    mat4 model;
};

void main() {
	tex_uv = uv;
	gl_Position = proj * view * model * vec4(pos, 1);
}