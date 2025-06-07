#version 450

layout(location = 0) out vec2 tex_uv;

const vec2 verts[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    tex_uv = verts[gl_VertexIndex] * 0.5 + 0.5;
    gl_Position = vec4(verts[gl_VertexIndex], 0.0, 1.0);
    // tex_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    // gl_Position = vec4(tex_uv * 2.0 - 1.0, 0.0, 1.0);
}