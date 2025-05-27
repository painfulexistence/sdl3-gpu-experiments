#pragma once
#include "SDL3/SDL.h"
#include "SDL3/SDL_gpu.h"

#include <vector>
#include <array>


struct PositionTextureVertex {
    float x, y, z;
    float u, v;
};
struct Vertex {
    float x, y, z, nx, ny, nz, u, v;
};

std::array<Vertex, 24> CreateCubeVertices();
std::array<Uint16, 36> CreateCubeIndices();
std::array<Vertex, 266> CreateSphereVertices();
std::array<Uint16, 1584> CreateSphereIndices();

class CPUMesh {
public:
    static CPUMesh CreateQuad();
    static CPUMesh CreateCube();

    CPUMesh(const std::vector<PositionTextureVertex>& vertices, const std::vector<Uint16>& indices)
        : _vertices(vertices), _indices(indices) {}
    ~CPUMesh() {}

    CPUMesh& Combine(const CPUMesh& mesh);

    template<typename... Meshes>
    CPUMesh& Combine(const Meshes&... meshes) {
        (Combine(meshes), ...);
        return *this;
    }

    size_t vertex_count() const { return _vertices.size(); }
    size_t index_count() const { return _indices.size(); }
    size_t vertex_byte_count() const { return sizeof(PositionTextureVertex); }
    size_t index_byte_count() const { return sizeof(Uint16); }
    bool has_indices() const { return !_indices.empty(); }

    const PositionTextureVertex* vertex_data() const { return _vertices.data(); }
    const Uint16* index_data() const { return _indices.data(); }

    void Stage(SDL_GPUDevice* device, SDL_GPUTransferBuffer* transferBuffer);
    void Upload(SDL_GPUDevice* device, SDL_GPUCopyPass* copyPass, SDL_GPUTransferBuffer* transferBuffer);
    void Bind(SDL_GPURenderPass* renderPass);
    void Draw(SDL_GPURenderPass* renderPass);
    void Release(SDL_GPUDevice* device);

private:
    std::vector<PositionTextureVertex> _vertices;
    std::vector<Uint16> _indices;

    SDL_GPUBuffer* _vertexBuffer = nullptr;
    SDL_GPUBuffer* _indexBuffer = nullptr;
};