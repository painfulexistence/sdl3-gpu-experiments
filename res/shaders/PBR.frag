#version 450

layout(location = 0) in vec2 tex_uv;
layout(location = 1) in vec3 frag_pos;
layout(location = 2) in vec3 frag_normal;
layout(location = 3) in vec3 frag_tangent;
layout(location = 4) in vec3 frag_bitangent;

layout(location = 0) out vec4 Color;

layout(set = 2, binding = 0) uniform sampler2D albedoMap;
layout(set = 2, binding = 1) uniform sampler2D normalMap;
layout(set = 2, binding = 2) uniform sampler2D metallicRoughnessMap;
layout(set = 2, binding = 3) uniform sampler2D occlusionMap;
layout(set = 2, binding = 4) uniform sampler2D emissiveMap;

layout(set = 3, binding = 0) uniform UBO {
    vec3 camera_pos;
};

struct Surface {
    vec3 color;
    float ao;
    float roughness;
    float metallic;
};

struct DirLight {
    vec3 direction;
    vec3 color;
    float intensity;
};

struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
};

const float PI = 3.1415927;
const float gamma = 2.2;
DirLight mainLight = { vec3(-20.0, 1.0, -5.0), vec3(1.0, 1.0, 1.0), 2.0 };
PointLight auxLights[] = {
    { vec3(0.0, -1.0, 0.0), vec3(1.0, 1.0, 1.0), 0.2 },
    { vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), 5.2 },
};

vec3 CookTorranceBRDF(vec3 norm, vec3 lightDir, vec3 viewDir, Surface surf);

float TrowbridgeReitzGGX(float nh, float r);

float SmithsSchlickGGX(float nv, float nl, float r);

vec3 FresnelSchlick(float nh, vec3 f0);

vec3 CalculateDirectionalLight(DirLight light, vec3 norm, vec3 viewDir, Surface surf) {
    vec3 lightDir = normalize(-light.direction);
    vec3 radiance = light.color * light.intensity;
    return CookTorranceBRDF(norm, lightDir, viewDir, surf) * radiance * clamp(dot(norm, lightDir), 0.0, 1.0);
}

vec3 CalculatePointLight(PointLight light, vec3 norm, vec3 viewDir, Surface surf) {
    vec3 lightDir = normalize(light.position - frag_pos);
    float dist = distance(light.position, frag_pos);
    float attenuation = 1.0 / (dist * dist);
    vec3 radiance = attenuation * light.color * light.intensity;
    return CookTorranceBRDF(norm, lightDir, viewDir, surf) * radiance * clamp(dot(norm, lightDir), 0.0, 1.0);
}

// vec3 CalculateIBL(vec3 norm, vec3 viewDir, Surface surf) {
//     vec3 reflectDir = reflect(-viewDir, norm);
//     float theta = -acos(reflectDir.y);
//     float phi = atan(reflectDir.z, reflectDir.x);
//     vec2 uv = fract(vec2(phi, theta) / vec2(2.0 * PI, PI) + vec2(0.5, 0.0));
//     vec3 env = texture(env_map, uv).rgb;
//     return env * surf.color;
// }

vec3 CookTorranceBRDF(vec3 norm, vec3 lightDir, vec3 viewDir, Surface surf) {
    vec3 halfway = normalize(lightDir + viewDir);
    float nv = max(dot(norm, viewDir), 0.0);
    float nl = max(dot(norm, lightDir), 0.0);
    float nh = max(dot(norm, halfway), 0.0);
    float vh = max(dot(viewDir, halfway), 0.0);

    float D = TrowbridgeReitzGGX(nh, surf.roughness + 0.01);
    float G = SmithsSchlickGGX(nv, nl, surf.roughness + 0.01);
    vec3 F = FresnelSchlick(vh, mix(vec3(0.04), surf.color, surf.metallic));

    vec3 specular = D * F * G / max(4.0 * nv * nl, 0.0001);
    vec3 kd = (1.0 - surf.metallic) * (vec3(1.0) - F);
    vec3 diffuse = kd * surf.color / PI;

    return diffuse + specular;
}

float TrowbridgeReitzGGX(float nh, float r) {
    float r2 = r * r;
    float a2 = r2 * r2;
    float nh2 = nh * nh;
    float nhr2 = (nh2 * (a2 - 1) + 1) * (nh2 * (a2 - 1) + 1);
    return a2 / (PI * nhr2);
}

float SmithsSchlickGGX(float nv, float nl, float r) {
    float k = (r + 1.0) * (r + 1.0) / 8.0;
    float ggx1 = nv / (nv * (1.0 - k) + k);
    float ggx2 = nl / (nl * (1.0 - k) + k);
    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float vh, vec3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
}

void main() {
    vec3 albedo = texture(albedoMap, tex_uv).rgb;
    vec3 normal = texture(normalMap, tex_uv).rgb * 2.0 - 1.0;
    vec3 metallicRoughness = texture(metallicRoughnessMap, tex_uv).rgb;
    float metallic = metallicRoughness.b;
    float roughness = metallicRoughness.g;
    float ao = texture(occlusionMap, tex_uv).r;
    vec3 emissive = texture(emissiveMap, tex_uv).rgb;

    Surface surf = {
        albedo,
        ao,
        roughness,
        metallic
    };

    mat3 TBN = mat3(
        normalize(frag_tangent),
        normalize(frag_bitangent),
        -normalize(frag_normal)
    );
    // TODO: normal mapping
    vec3 norm = normalize(frag_normal);
    vec3 viewDir = normalize(camera_pos - frag_pos);

    vec3 result = vec3(0.0);
    result += CalculateDirectionalLight(mainLight, norm, viewDir, surf);
    // result += CalculatePointLight(auxLights[0], norm, viewDir, surf);
    result += vec3(0.1) * surf.ao * surf.color;
    // result += CalculateIBL(norm, viewDir, surf);

    result = pow(result, vec3(1.0 / gamma));

    Color = vec4(result, 1.0);
}
