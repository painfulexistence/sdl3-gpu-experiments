#version 450

// GPU-driven multi-draw indirect vertex shader.
// This is a fully vertex-pulling shader: no vertex input state is bound. It runs
// under a non-indexed indirect draw where vertexCount == the mesh's index count,
// so gl_VertexIndex walks the mesh's index range and gl_InstanceIndex (already
// offset by the command's firstInstance) walks the compacted instance range.

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

layout(std430, set = 0, binding = 0) readonly buffer InstanceBuffer {
    Instance instances[];
};

layout(std430, set = 0, binding = 1) readonly buffer MeshInfoBuffer {
    MeshInfo meshes[];
};

// Raw float view of the vertex buffer. The CPU Vertex is a flat 8-float struct
// { x, y, z, nx, ny, nz, u, v } (32-byte stride), so we index by hand instead of
// relying on std430 vec3 padding (which would wrongly assume a 48-byte stride).
layout(std430, set = 0, binding = 2) readonly buffer VertexBuffer {
    float vertexData[];
};

layout(std430, set = 0, binding = 3) readonly buffer IndexBuffer {
    uint indices[];
};

layout(std430, set = 0, binding = 4) readonly buffer CompactedInstanceBuffer {
    uint compactedInstances[];
};

void main() {
    uint instIdx = compactedInstances[gl_InstanceIndex];
    Instance inst = instances[instIdx];
    MeshInfo mesh = meshes[inst.meshID];
    materialID = inst.materialID;

    uint index = indices[mesh.baseIndex + uint(gl_VertexIndex)];
    uint v = (uint(mesh.baseVertex) + index) * 8u;

    vec3 position = vec3(vertexData[v + 0u], vertexData[v + 1u], vertexData[v + 2u]);
    vec3 normal   = vec3(vertexData[v + 3u], vertexData[v + 4u], vertexData[v + 5u]);
    vec2 uv       = vec2(vertexData[v + 6u], vertexData[v + 7u]);

    frag_pos = vec3(inst.model * vec4(position, 1.0));
    frag_normal = vec3(inst.model * vec4(normal, 0.0));
    tex_uv = uv;
    gl_Position = proj * view * vec4(frag_pos, 1.0);
}
