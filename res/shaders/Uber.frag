#version 450

layout(location = 0) in vec2 tex_uv;
layout(location = 1) in vec3 frag_pos;
layout(location = 2) in vec3 frag_normal;
layout(location = 3) in flat uint materialID;

layout(location = 0) out vec4 color;

struct Material {
    vec3 diffuse;
    vec3 specular;
    float roughness;
};

layout(std430, set = 2, binding = 0) readonly buffer MaterialBuffer {
    Material materials[];
};

layout(set = 3, binding = 0) uniform UBO {
    vec3 camera_pos;
};

void main() {
    Material mat = materials[materialID];
    vec3 diffuseColor = mat.diffuse;
    vec3 specularColor = mat.specular;
    float roughness = mat.roughness;

    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float lightIntensity = 1.0;

    vec3 normal = normalize(frag_normal);
    vec3 viewDir = normalize(camera_pos - frag_pos);
    vec3 halfDir = normalize(lightDir + viewDir);
    float nh = max(dot(normal, halfDir), 0.0);
    float nl = max(dot(normal, lightDir), 0.0);
    float nv = max(dot(normal, viewDir), 0.0);

    vec3 diffuse = nl * diffuseColor;
    float spec = pow(nh, 32.0 * (1.0 - roughness));
    vec3 specular = spec * 0.5 * specularColor;
    vec3 ambient = 0.1 * diffuseColor;

    vec3 result = (ambient + diffuse + specular) * lightColor * lightIntensity;

	color = vec4(result, 1.0);
}