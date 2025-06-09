#pragma once

#include "SDL3/SDL_gpu.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

struct Image {
public:
    std::string uri;
    Uint32 width;
    Uint32 height;
    Uint32 component;
    std::vector<Uint8> pixels;
    Uint32 num_levels = 1;
    std::unique_ptr<SDL_GPUTexture, std::function<void(SDL_GPUTexture*)>> texture;

    Uint32 total_byte_count() { return pixels.size(); }

    void Prepare(SDL_GPUDevice* device);
    void Stage(SDL_GPUDevice* device, SDL_GPUTransferBuffer* transferBuffer, Uint32 offset = 0);
    void Upload(SDL_GPUDevice* device, SDL_GPUCopyPass* copyPass, SDL_GPUTransferBuffer* transferBuffer, Uint32 offset = 0);
    void Release(SDL_GPUDevice* device);

    void GenerateMipmaps(SDL_GPUCommandBuffer* cmd);
};