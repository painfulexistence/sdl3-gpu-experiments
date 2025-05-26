#pragma once
#include "SDL3/SDL.h"

#include <vector>
#include <array>


struct PositionTextureVertex {
    float x, y, z;
    float u, v;
};

class CPUMesh {
public:
    CPUMesh(const std::vector<PositionTextureVertex>& vertices, const std::vector<Uint16>& indices)
        : _vertices(vertices), _indices(indices) {}

    static CPUMesh CreateQuad();
    static CPUMesh CreateCube();

    size_t vertex_count() const { return _vertices.size(); }
    size_t index_count() const { return _indices.size(); }
    size_t vertex_byte_count() const { return sizeof(PositionTextureVertex); }
    size_t index_byte_count() const { return sizeof(Uint16); }
    bool has_indices() const { return !_indices.empty(); }

    const PositionTextureVertex* vertex_data() const { return _vertices.data(); }
    const Uint16* index_data() const { return _indices.data(); }

private:
    std::vector<PositionTextureVertex> _vertices;
    std::vector<Uint16> _indices;
};