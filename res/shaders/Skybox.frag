#version 450

// HDRI equirectangular skybox fragment shader. Reconstructs the per-pixel world
// view ray from the fullscreen-triangle NDC and the inverse camera matrices, then
// samples the equirectangular HDR environment map.

layout(location = 0) in vec2 ndc;
layout(location = 0) out vec4 color;

layout(set = 2, binding = 0) uniform sampler2D equirectMap;

layout(set = 3, binding = 0) uniform SkyboxParams {
    mat4 invProj;   // inverse projection
    mat4 invView;   // inverse view (rotation used; translation ignored for a skybox)
    float exposure;
};

const float PI = 3.14159265359;
const vec2 INV_ATAN = vec2(0.1591, 0.3183); // (1/2pi, 1/pi)

vec2 sampleSphericalMap(vec3 dir) {
    // atan(z, x) in [-pi, pi], asin(y) in [-pi/2, pi/2] -> [0,1] uv.
    vec2 uv = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));
    uv *= INV_ATAN;
    uv += 0.5;
    return uv;
}

void main() {
    // NDC (far plane) -> view space -> world direction.
    vec4 viewPos = invProj * vec4(ndc, 1.0, 1.0);
    vec3 viewDir = normalize(viewPos.xyz / viewPos.w);
    vec3 worldDir = normalize(mat3(invView) * viewDir);

    vec3 hdr = texture(equirectMap, sampleSphericalMap(worldDir)).rgb * exposure;
    color = vec4(hdr, 1.0);
}
