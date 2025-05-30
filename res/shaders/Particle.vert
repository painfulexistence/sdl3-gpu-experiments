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
    // FIXME: seems not so correct
    gl_Position = vec4(0.005 * pos + (proj* view * vec4(particle.position, 1.0)).xyz, 1.0);
    frag_color = particle.color.rgb;
    tex_uv = uv;
}
