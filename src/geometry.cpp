#include "geometry.hpp"
#include <glm/geometric.hpp>
#include <glm/vec3.hpp>
#include <glm/trigonometric.hpp>

std::array<PositionTextureVertex, 6> quad = {{
    // bottom-right
    { -1, -1, 0, 0, 0 },
    {  1, -1, 0, 1, 0 },
	{  1,  1, 0, 1, 1 },
    // top-left
	{ -1, -1, 0, 0, 0 },
	{  1,  1, 0, 1, 1 },
	{ -1,  1, 0, 0, 1 }
}};
// Same as:
// PositionTextureVertex quad[6] = {
//     { -1, -1, 0, 0, 0 },
// 	{  1, -1, 0, 1, 0 },
// 	{  1,  1, 0, 1, 1 },
// 	{ -1, -1, 0, 0, 0 },
// 	{  1,  1, 0, 1, 1 },
// 	{ -1,  1, 0, 0, 1 }
// };
std::array<PositionTextureVertex, 24> cube = {{
    // left
    { .5f, .5f, .5f, 1.0f, 1.0f },
    { .5f, -.5f, .5f, 1.0f, 0.0f },
    { -.5f, .5f, .5f, 0.0f, 1.0f },
    { -.5f, -.5f, .5f, 0.0f, 0.0f },
    // right
    { .5f, .5f, -.5f, 1.0f, 1.0f },
    { .5f, -.5f, -.5f, 1.0f, 0.0f },
    { -.5f, .5f, -.5f, 0.0f, 1.0f },
    { -.5f, -.5f, -.5f, 0.0f, 0.0f },
    // back
    { -.5f, .5f, .5f, 1.0f, 1.0f },
    { -.5f, .5f, -.5f, 0.0f, 1.0f },
    { -.5f, -.5f, .5f, 1.0f, 0.0f },
    { -.5f, -.5f, -.5f, 0.0f, 0.0f },
    // front
    { .5f, .5f, .5f, 1.0f, 1.0f },
    { .5f, .5f, -.5f, 0.0f, 1.0f },
    { .5f, -.5f, .5f, 1.0f, 0.0f },
    { .5f, -.5f, -.5f, 0.0f, 0.0f },
    // top
    { .5f, .5f, .5f, 1.0f, 1.0f },
    { .5f, .5f, -.5f, 1.0f, 0.0f },
    { -.5f, .5f, .5f, 0.0f, 1.0f },
    { -.5f, .5f, -.5f, 0.0f, 0.0f },
    // bottom
    { .5f, -.5f, .5f, 1.0f, 1.0f },
    { .5f, -.5f, -.5f, 1.0f, 0.0f },
    { -.5f, -.5f, .5f, 0.0f, 1.0f },
    { -.5f, -.5f, -.5f, 0.0f, 0.0f },
}};
std::array<Uint16, 36> cubeIndices = {{
    0,  2,  1,  1,  2,  3,  4,  5,  6,  6,  5,  7,  8,  9,  10, 10, 9,  11,
    12, 14, 13, 13, 14, 15, 16, 17, 18, 18, 17, 19, 20, 22, 21, 21, 22, 23
}};

CPUMesh CPUMesh::CreateQuad() {
    return CPUMesh({
        // bottom-right
        { -1, -1, 0, 0, 0 },
        {  1, -1, 0, 1, 0 },
        {  1,  1, 0, 1, 1 },
        // top-left
        { -1, -1, 0, 0, 0 },
        {  1,  1, 0, 1, 1 },
        { -1,  1, 0, 0, 1 }
    }, {});
}

CPUMesh CPUMesh::CreateCube() {
    return CPUMesh({
        // left
        { .5f, .5f, .5f, 1.0f, 1.0f },
        { .5f, -.5f, .5f, 1.0f, 0.0f },
        { -.5f, .5f, .5f, 0.0f, 1.0f },
        { -.5f, -.5f, .5f, 0.0f, 0.0f },
        // right
        { .5f, .5f, -.5f, 1.0f, 1.0f },
        { .5f, -.5f, -.5f, 1.0f, 0.0f },
        { -.5f, .5f, -.5f, 0.0f, 1.0f },
        { -.5f, -.5f, -.5f, 0.0f, 0.0f },
        // back
        { -.5f, .5f, .5f, 1.0f, 1.0f },
        { -.5f, .5f, -.5f, 0.0f, 1.0f },
        { -.5f, -.5f, .5f, 1.0f, 0.0f },
        { -.5f, -.5f, -.5f, 0.0f, 0.0f },
        // front
        { .5f, .5f, .5f, 1.0f, 1.0f },
        { .5f, .5f, -.5f, 0.0f, 1.0f },
        { .5f, -.5f, .5f, 1.0f, 0.0f },
        { .5f, -.5f, -.5f, 0.0f, 0.0f },
        // top
        { .5f, .5f, .5f, 1.0f, 1.0f },
        { .5f, .5f, -.5f, 1.0f, 0.0f },
        { -.5f, .5f, .5f, 0.0f, 1.0f },
        { -.5f, .5f, -.5f, 0.0f, 0.0f },
        // bottom
        { .5f, -.5f, .5f, 1.0f, 1.0f },
        { .5f, -.5f, -.5f, 1.0f, 0.0f },
        { -.5f, -.5f, .5f, 0.0f, 1.0f },
        { -.5f, -.5f, -.5f, 0.0f, 0.0f },
    }, {
        0,  2,  1,  1,  2,  3,  4,  5,  6,  6,  5,  7,  8,  9,  10, 10, 9,  11,
        12, 14, 13, 13, 14, 15, 16, 17, 18, 18, 17, 19, 20, 22, 21, 21, 22, 23
    });
}

CPUMesh& CPUMesh::Combine(const CPUMesh& mesh) {
    _vertices.insert(_vertices.end(), mesh._vertices.begin(), mesh._vertices.end());
    _indices.insert(_indices.end(), mesh._indices.begin(), mesh._indices.end());
    return *this;
}

void CPUMesh::Stage(SDL_GPUDevice* device, SDL_GPUTransferBuffer* transferBuffer) {
    // Create buffers
    SDL_GPUBufferCreateInfo vertexBufferCreateInfo = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = static_cast<Uint32>(vertex_byte_count() * vertex_count())
    };
    _vertexBuffer = SDL_CreateGPUBuffer(device, &vertexBufferCreateInfo);

    if (has_indices()) {
        SDL_GPUBufferCreateInfo indexBufferCreateInfo = {
            .usage = SDL_GPU_BUFFERUSAGE_INDEX,
            .size = static_cast<Uint32>(index_byte_count() * index_count())
        };
        _indexBuffer = SDL_CreateGPUBuffer(device, &indexBufferCreateInfo);
    }

    // Copy data to transfer buffer
	PositionTextureVertex* transferData = reinterpret_cast<PositionTextureVertex*>(
        SDL_MapGPUTransferBuffer(
            device,
            transferBuffer,
            false
        )
	);
    memcpy(transferData, vertex_data(), vertex_byte_count() * vertex_count());
    if (has_indices()) {
        memcpy((Uint16*)&transferData[vertex_count()], index_data(),  index_byte_count() * index_count());
    }
	SDL_UnmapGPUTransferBuffer(device, transferBuffer);
}

void CPUMesh::Upload(SDL_GPUDevice* device, SDL_GPUCopyPass* copyPass, SDL_GPUTransferBuffer* transferBuffer) {
    SDL_GPUTransferBufferLocation bufTransferInfo = {
        .transfer_buffer = transferBuffer,
        .offset = 0
    };
    SDL_GPUBufferRegion bufTransferRegion = {
        .buffer = _vertexBuffer,
        .offset = 0,
        .size = static_cast<Uint32>(vertex_byte_count() * vertex_count())
    };
	SDL_UploadToGPUBuffer(
		copyPass,
		&bufTransferInfo,
		&bufTransferRegion,
		false
	);
    if (has_indices()) {
        bufTransferInfo.offset = vertex_byte_count() * vertex_count();
        bufTransferRegion.buffer = _indexBuffer;
        bufTransferRegion.size = static_cast<Uint32>(index_byte_count() * index_count());
        SDL_UploadToGPUBuffer(
            copyPass,
            &bufTransferInfo,
            &bufTransferRegion,
            false
        );
    }
}

void CPUMesh::Bind(SDL_GPURenderPass* renderPass) {
    SDL_GPUBufferBinding vertexBufferBinding = { .buffer = _vertexBuffer, .offset = 0 };
    SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBufferBinding, 1);
    if (has_indices()) {
        SDL_GPUBufferBinding indexBufferBinding = { .buffer = _indexBuffer, .offset = 0 };
        SDL_BindGPUIndexBuffer(renderPass, &indexBufferBinding, SDL_GPU_INDEXELEMENTSIZE_16BIT);
    }
}

void CPUMesh::Draw(SDL_GPURenderPass* renderPass) {
    if (has_indices()) {
        SDL_DrawGPUIndexedPrimitives(renderPass, index_count(), 1, 0, 0, 0);
    } else {
        SDL_DrawGPUPrimitives(renderPass, vertex_count(), 1, 0, 0);
    }
}

void CPUMesh::Release(SDL_GPUDevice* device) {
    if (_vertexBuffer) {
        SDL_ReleaseGPUBuffer(device, _vertexBuffer);
    }
    if (_indexBuffer) {
        SDL_ReleaseGPUBuffer(device, _indexBuffer);
    }
}

std::array<Vertex, 24> CreateCubeVertices() {
    return std::array<Vertex, 24>({{
        // front
        { .5f, .5f, .5f, 0.0f, 0.0f, 1.0f, 1.f, 1.f },
        { -.5f, .5f, .5f, 0.0f, 0.0f, 1.0f, 0.f, 1.f },
        { .5f, -.5f, .5f, 0.0f, 0.0f, 1.0f, 1.f, 0.f },
        { -.5f, -.5f, .5f, 0.0f, 0.0f, 1.0f, 0.f, 0.f },
        // back
        { -.5f, .5f, -.5f, 0.0f, 0.0f, -1.0f, 1.f, 1.f },
        { .5f, .5f, -.5f, 0.0f, 0.0f, -1.0f, 0.f, 1.f },
        { -.5f, -.5f, -.5f, 0.0f, 0.0f, -1.0f, 1.f, 0.f },
        { .5f, -.5f, -.5f, 0.0f, 0.0f, -1.0f, 0.f, 0.f },
        // right
        { .5f, .5f, -.5f, 1.0f, 0.0f, 0.0f, 1.f, 1.f },
        { .5f, .5f, .5f, 1.0f, 0.0f, 0.0f, 0.f, 1.f },
        { .5f, -.5f, -.5f, 1.0f, 0.0f, 0.0f, 1.f, 0.f },
        { .5f, -.5f, .5f, 1.0f, 0.0f, 0.0f, 0.f, 0.f },
        // left
        { -.5f, .5f, .5f, -1.0f, 0.0f, 0.0f, 1.f, 1.f },
        { -.5f, .5f, -.5f, -1.0f, 0.0f, 0.0f, 0.f, 1.f },
        { -.5f, -.5f, .5f, -1.0f, 0.0f, 0.0f, 1.f, 0.f },
        { -.5f, -.5f, -.5f, -1.0f, 0.0f, 0.0f, 0.f, 0.f },
        // top
        { .5f, .5f, -.5f, 0.0f, 1.0f, 0.0f, 1.f, 1.f },
        { -.5f, .5f, -.5f, 0.0f, 1.0f, 0.0f, 0.f, 1.f },
        { .5f, .5f, .5f, 0.0f, 1.0f, 0.0f, 1.f, 0.f },
        { -.5f, .5f, .5f, 0.0f, 1.0f, 0.0f, 0.f, 0.f },
        // bottom
        { .5f, -.5f, -.5f, 0.0f, -1.0f, 0.0f, 1.f, 1.f },
        { -.5f, -.5f, -.5f, 0.0f, -1.0f, 0.0f, 0.f, 1.f },
        { .5f, -.5f, -.5f, 0.0f, -1.0f, 0.0f, 1.f, 0.f },
        { -.5f, -.5f, .5f, 0.0f, -1.0f, 0.0f, 0.f, 0.f }
    }});
}

std::array<Uint32, 36> CreateCubeIndices() {
    return std::array<Uint32, 36>({
        0, 1, 2,
        2, 1, 3,
        4, 5, 6,
        6, 5, 7,
        8, 9, 10,
        10, 9, 11,
        12, 13, 14,
        14, 13, 15,
        16, 17, 18,
        18, 17, 19,
        20, 21, 22,
        22, 21, 23
    });
}

std::array<Vertex, 266> CreateSphereVertices() {
    float delta = M_PI / (float)12;

    std::array<Vertex, 266> verts;
    verts[0] = Vertex{ 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f };
    for (int v = 1; v <= 11; ++v) {
        float vAngle = v * delta;
        for (int h = 0; h <= 23; ++h) {
            float hAngle = h * delta;
            glm::vec3 pos = glm::vec3(
                glm::sin(vAngle) * glm::cos(hAngle),
                glm::cos(vAngle),
                glm::sin(vAngle) * glm::sin(hAngle)
            );
            glm::vec3 norm = glm::normalize(pos);
            verts[(v - 1) * 24 + h + 1] = Vertex{
                pos.x, pos.y, pos.z,
                norm.x, norm.y, norm.z,
                (float)h / (float)24, 1.0f - (float)v / (float)12
            };
        }
    }
    verts[265] = Vertex{ 0.f, -1.f, 0.f, 0.f, -1.f, 0.f, 0.f, 0.f };

    return verts;
}

std::array<Uint32, 1584> CreateSphereIndices() {
    std::array<Uint32, 1584> tris;

    for (int h = 0; h <= 23; ++h) {
        tris[h * 3] = 0;
        tris[h * 3 + 1] = h + 1;
        tris[h * 3 + 2] = (h + 2) % 24;
    }
    for (int v = 1; v <= 10; ++v) {
        for (int h = 0; h <= 23; ++h) {
            // top-left triangles
            tris[(v - 1) * 144 + h * 6 + 73] = (v - 1) * 24 + h + 1;
            tris[(v - 1) * 144 + h * 6 + 74] = v * 24 + (h + 1) % 24;
            tris[(v - 1) * 144 + h * 6 + 75] = (v - 1) * 24 + (h + 1) % 24 + 1;
            // bottom-right triangles
            tris[(v - 1) * 144 + h * 6 + 76] = (v - 1) * 24 + (h + 1) % 24 + 1;
            tris[(v - 1) * 144 + h * 6 + 77] = v * 24 + (h + 1) % 24;
            tris[(v - 1) * 144 + h * 6 + 78] = v * 24 + (h + 1) % 24 + 1;
        }
    }
    for (int h = 0; h <= 23; ++h) {
        tris[h * 3 + 1512] = 240 + h + 1;
        tris[h * 3 + 1513] = 265;
        tris[h * 3 + 1514] = 240 + (h + 1) % 24 + 1;
    }
    return tris;
}