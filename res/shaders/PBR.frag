#version 450

layout(location = 0) in vec2 tex_uv;
layout(location = 1) in vec3 frag_normal;

layout(location = 0) out vec4 Color;

layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D normalMap;
layout(set = 2, binding = 2) uniform sampler2D metallicRoughnessMap;
layout(set = 2, binding = 3) uniform sampler2D occlusionMap;
layout(set = 2, binding = 4) uniform sampler2D emissiveMap;


void main() {

    vec3 albedo = texture(albedoMap, tex_uv).rgb;
    // vec3 normal = texture(normalMap, tex_uv).rgb;
    // vec3 metallicRoughness = texture(metallicRoughnessMap, tex_uv).rgb;
    // vec3 occlusion = texture(occlusionMap, tex_uv).rgb;
    // vec3 emissive = texture(emissiveMap, tex_uv).rgb;

    vec3 norm = normalize(frag_normal);
    vec3 lightDir = normalize(vec3(20.0, -1.0, 5.0));


    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * albedo * 2.0;
    vec3 ambient = vec3(0.01, 0.01, 0.01);

    vec3 result = (ambient + diffuse) * vec3(1.0, 1.0, 1.0);
    Color = vec4(result, 1.0);
}