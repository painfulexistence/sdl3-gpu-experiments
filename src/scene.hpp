#pragma once
#include "SDL3/SDL_gpu.h"
#include <glm/mat4x4.hpp>
#include <string>
#include <vector>
#include <memory>

struct Material {
    std::shared_ptr<SDL_GPUTexture> albedoMap;
    std::shared_ptr<SDL_GPUTexture> normalMap;
    std::shared_ptr<SDL_GPUTexture> metallicRoughnessMap;
    std::shared_ptr<SDL_GPUTexture> occlusionMap;
    std::shared_ptr<SDL_GPUTexture> emissiveMap;
    std::shared_ptr<SDL_GPUGraphicsPipeline> pipeline = nullptr;
    SDL_GPUGraphicsPipelineCreateInfo pipelineInfo;

    std::shared_ptr<SDL_GPUGraphicsPipeline> GetPipeline(SDL_GPUDevice* device, SDL_GPUTextureFormat renderTargetFormat, SDL_GPUSampleCount msaaSampleCount) {
        if (pipeline == nullptr) {
            pipelineInfo.target_info = {
                .color_target_descriptions = (SDL_GPUColorTargetDescription[]){{
                    .format = renderTargetFormat
                }},
                .num_color_targets = 1,
            };
            pipelineInfo.multisample_state = {
                .sample_count = msaaSampleCount
            };
            pipeline = std::shared_ptr<SDL_GPUGraphicsPipeline>(
                SDL_CreateGPUGraphicsPipeline(device, &pipelineInfo),
                [device](SDL_GPUGraphicsPipeline* p) {
                    SDL_ReleaseGPUGraphicsPipeline(device, p);
                }
            );
        }
        return pipeline;
    }
};

struct SubMesh {
    std::vector<std::shared_ptr<SDL_GPUBuffer>> vbos;
    std::vector<SDL_GPUVertexBufferDescription> vertexBufferDescs;
    std::vector<SDL_GPUVertexAttribute> vertexAttributes;
    std::shared_ptr<SDL_GPUBuffer> ebo = nullptr;
    std::shared_ptr<Material> material = nullptr;
    size_t bufferSize = 0;
    size_t vertexCount = 0;
    size_t indexCount = 0;
    SDL_GPUIndexElementSize indexType;
};

struct Mesh {
    std::string name;
    std::vector<std::unique_ptr<SubMesh>> subMeshes;
};

struct Node {
    std::string name;
    std::vector<std::shared_ptr<Node>> children;
    glm::mat4 localTransform;
    std::shared_ptr<Mesh> mesh = nullptr;

    void SetLocalTransform(const glm::mat4& transform) {
        localTransform = transform;
    }

    std::shared_ptr<Node> CreateChild(const std::string& name, const glm::mat4& localTransform) {
        auto child = std::make_shared<Node>();
        child->name = name;
        child->localTransform = localTransform;
        children.push_back(child);
        return child;
    }
    void AddChild(std::shared_ptr<Node> child) {
        children.push_back(child);
    }
};

class Scene {
public:
    std::string name;
    std::vector<std::shared_ptr<Node>> nodes;

    Scene() = default;
    Scene(const std::string& name) : name(name) {};
    ~Scene() = default;

    void Print();

    std::shared_ptr<Node> CreateNode(const std::string& name, const glm::mat4& transform);
    void AddNode(std::shared_ptr<Node> node);
    std::shared_ptr<Node> FindNode(const std::string& name);
    std::shared_ptr<Node> FindNodeInHierarchy(const std::string& name, const std::shared_ptr<Node>& node);
    void Update(float dt);
    void Draw();

    void Release(SDL_GPUDevice* device);
};