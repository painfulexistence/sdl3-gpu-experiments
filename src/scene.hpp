#pragma once
#include "SDL3/SDL_gpu.h"
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <vector>
#include <memory>

struct Image {
    std::string uri;
    Uint32 width;
    Uint32 height;
    Uint32 component;
    std::vector<Uint8> pixels;
    std::unique_ptr<SDL_GPUTexture, std::function<void(SDL_GPUTexture*)>> texture;
};

struct Material {
    std::string name;
    std::shared_ptr<Image> albedoMap;
    std::shared_ptr<Image> normalMap;
    std::shared_ptr<Image> metallicRoughnessMap;
    std::shared_ptr<Image> occlusionMap;
    std::shared_ptr<Image> emissiveMap;
    SDL_GPUGraphicsPipelineCreateInfo pipelineInfo;
    std::shared_ptr<SDL_GPUGraphicsPipeline> pipeline;

    // std::shared_ptr<SDL_GPUGraphicsPipeline> GetPipeline(SDL_GPUDevice* device, SDL_GPUTextureFormat renderTargetFormat, SDL_GPUSampleCount msaaSampleCount) {
    //     if (pipeline == nullptr) {
    //         pipelineInfo.target_info = {
    //             .color_target_descriptions = (SDL_GPUColorTargetDescription[]){{
    //                 .format = renderTargetFormat
    //             }},
    //             .num_color_targets = 1,
    //         };
    //         pipelineInfo.multisample_state = {
    //             .sample_count = msaaSampleCount
    //         };
    //         pipeline = std::shared_ptr<SDL_GPUGraphicsPipeline>(
    //             SDL_CreateGPUGraphicsPipeline(device, &pipelineInfo),
    //             [device](SDL_GPUGraphicsPipeline* p) {
    //                 SDL_ReleaseGPUGraphicsPipeline(device, p);
    //             }
    //         );
    //     }
    //     return pipeline;
    // }
};

enum class PrimitiveMode {
    POINTS,
    LINES,
    LINE_STRIP,
    TRIANGLES,
    TRIANGLE_STRIP,
};

struct Mesh {
    std::vector<std::shared_ptr<SDL_GPUBuffer>> vbos;
    std::vector<SDL_GPUVertexBufferDescription> vertexBufferDescs;
    std::vector<SDL_GPUVertexAttribute> vertexAttributes;
    std::shared_ptr<SDL_GPUBuffer> ebo = nullptr;
    size_t bufferSize = 0;
    size_t vertexCount = 0;
    size_t indexCount = 0;
    SDL_GPUIndexElementSize indexType;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uv0s;
    std::vector<glm::vec2> uv1s;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec4> colors;
    std::vector<Uint32> indices;
    std::shared_ptr<Material> material = nullptr;
    PrimitiveMode primitiveMode;
};

struct MeshGroup {
    std::string name;
    std::vector<std::unique_ptr<Mesh>> meshes;
};

struct Node {
    std::string name;
    std::vector<std::shared_ptr<Node>> children;
    glm::mat4 localTransform;
    glm::mat4 worldTransform; // calculated from localTransform and parent's worldTransform
    std::shared_ptr<MeshGroup> meshGroup = nullptr;
    bool isTransformDirty = true;

    void SetLocalTransform(const glm::mat4& transform) {
        localTransform = transform;
        isTransformDirty = true;
    }

    std::shared_ptr<Node> CreateChild(const std::string& name, const glm::mat4& localTransform) {
        auto child = std::make_shared<Node>();
        child->name = name;
        child->localTransform = localTransform;
        child->isTransformDirty = true;
        children.push_back(child);
        return child;
    }
    void AddChild(std::shared_ptr<Node> child) {
        child->isTransformDirty = true;
        children.push_back(child);
    }
};

class Scene {
public:
    std::string name;
    std::vector<std::shared_ptr<Image>> images;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Node>> nodes;

    Scene() = default;
    Scene(const std::string& name) : name(name) {};
    ~Scene() = default;

    void Print();

    void Upload(SDL_GPUDevice* device);
    void Update(float dt);
    void Draw(SDL_GPURenderPass* renderPass);
    void Release(SDL_GPUDevice* device);

    std::shared_ptr<Node> CreateNode(const std::string& name, const glm::mat4& transform);
    void AddNode(std::shared_ptr<Node> node);
    std::shared_ptr<Node> FindNode(const std::string& name);
    std::shared_ptr<Node> FindNodeInHierarchy(const std::string& name, const std::shared_ptr<Node>& node);

private:
    void PrintNode(const std::shared_ptr<Node>& node);

    void UploadNode(const std::shared_ptr<Node>& node, SDL_GPUDevice* device);
    void UpdateNode(const std::shared_ptr<Node>& node, const glm::mat4& parentTransform);
    void ReleaseNode(const std::shared_ptr<Node>& node, SDL_GPUDevice* device);

    auto CreateTexture(const std::shared_ptr<Image>& image, SDL_GPUDevice* device) -> std::unique_ptr<SDL_GPUTexture, std::function<void(SDL_GPUTexture*)>>;
    auto CreateBuffer(const void* data, size_t size, SDL_GPUBufferUsageFlags usage, SDL_GPUDevice* device) -> std::shared_ptr<SDL_GPUBuffer>;
};