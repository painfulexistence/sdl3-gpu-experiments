#version 450

// HDRI equirectangular skybox — fullscreen triangle. Emits clip-space NDC that
// covers the screen at the far plane; the fragment shader reconstructs a world
// ray from the NDC and samples the equirect map. No vertex buffer is bound.

layout(location = 0) out vec2 ndc;

void main() {
    // Fullscreen triangle from gl_VertexIndex (0,1,2).
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    ndc = uv * 2.0 - 1.0;
    // z = 1.0 -> far plane, so the skybox stays behind everything (depth compare
    // LESS_OR_EQUAL, no depth write).
    gl_Position = vec4(ndc, 1.0, 1.0);
}
