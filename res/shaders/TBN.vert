#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;

layout(location = 0) out vec2 tex_uv;
layout(location = 1) out vec3 frag_pos;
layout(location = 2) out vec3 frag_normal;
layout(location = 3) out vec3 frag_tangent;
layout(location = 4) out vec3 frag_bitangent;

layout(set = 1, binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
};
layout(set = 1, binding = 1) uniform Instance {
    mat4 model;
};

void main() {
	tex_uv = uv;
    frag_pos = (model * vec4(pos, 1.0)).xyz;
    frag_normal = normalize((model * vec4(normal, 0.0)).xyz);
    frag_tangent = normalize((model * vec4(tangent, 0.0)).xyz);
    frag_bitangent = normalize(cross(frag_normal, frag_tangent));
	gl_Position = proj * view * vec4(frag_pos, 1.0);
}