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
};

struct SubMesh {
    SDL_GPUPrimitiveType mode = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
    std::vector<std::shared_ptr<SDL_GPUBuffer>> vbos;
    std::shared_ptr<SDL_GPUBuffer> ebo = nullptr;
    std::shared_ptr<Material> material;
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
};

struct Scene {
    std::string name;
    std::vector<std::shared_ptr<Node>> nodes;
};