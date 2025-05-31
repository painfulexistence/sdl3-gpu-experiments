#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 tex_uv;

layout(set = 1, binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
};

struct Particle {
    vec3 position;
    vec3 velocity;
    vec3 force;
    vec4 color;
};

layout(std140, set = 0, binding = 0) readonly buffer ParticleBuffer {
    Particle particles[];
};

void main() {
    Particle particle = particles[gl_InstanceIndex];
    vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 up = vec3(view[0][1], view[1][1], view[2][1]);
    vec3 vert_pos = particle.position + right * pos.x * 0.05 + up * pos.y * 0.05;
    gl_Position = proj * view * vec4(vert_pos, 1.0);
    frag_color = particle.color.rgb;
    tex_uv = uv;
}
