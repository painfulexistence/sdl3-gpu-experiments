#version 450

layout(location = 0) out vec2 tex_uv;
layout(location = 1) out vec3 frag_pos;
layout(location = 2) out vec3 frag_normal;
layout(location = 3) out flat uint materialID;

layout(set = 1, binding = 0) uniform CameraInfo {
    mat4 view;
    mat4 proj;
};

struct Instance {
    mat4 model;
    uint meshID;
    uint materialID;
};

struct MeshInfo {
    int baseVertex;
    uint baseIndex;
    uint indexCount;
};

struct MaterialInfo {
    vec3 diffuse;
    vec3 specular;
    float roughness;
};

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
};

layout(std430, set = 0, binding = 0) readonly buffer InstanceBuffer {
    Instance instances[];
};

layout(std430, set = 0, binding = 1) readonly buffer MeshInfoBuffer {
    MeshInfo meshes[];
};

layout(std430, set = 0, binding = 2) readonly buffer MaterialInfoBuffer {
    MaterialInfo materials[];
};

layout(std430, set = 0, binding = 3) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(std430, set = 0, binding = 4) readonly buffer IndexBuffer {
    uint indices[];
};

void main() {
	uint meshID = instances[gl_InstanceIndex].meshID;
    materialID = instances[gl_InstanceIndex].materialID;
    MeshInfo mesh = meshes[meshID];
    uint index = indices[mesh.baseIndex];
    vec3 pos = vertices[mesh.baseVertex + index].position;
    vec3 normal = vertices[mesh.baseVertex + index].normal;
    vec2 uv = vertices[mesh.baseVertex + index].uv;
	frag_pos = vec3(instances[gl_InstanceIndex].model * vec4(pos, 1.0));
	frag_normal = vec3(instances[gl_InstanceIndex].model * vec4(normal, 0.0));
	tex_uv = uv;
	gl_Position = proj * view * vec4(frag_pos, 1.0);
}