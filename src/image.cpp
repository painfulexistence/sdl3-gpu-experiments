#include "image.hpp"
#include <SDL3/SDL.h>

void Image::Prepare(SDL_GPUDevice* device) {
    num_levels = std::max(1u, static_cast<Uint32>(std::floor(std::log2(std::min(width, height)))));

    SDL_GPUTextureCreateInfo textureDesc = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = static_cast<Uint32>(width),
        .height = static_cast<Uint32>(height),
        .layer_count_or_depth = 1,
        .num_levels = num_levels,
    };
    if (component == 1) {
        textureDesc.format = SDL_GPU_TEXTUREFORMAT_R8_UNORM;
    } else if (component == 2) {
        textureDesc.format = SDL_GPU_TEXTUREFORMAT_R8G8_UNORM;
    } else if (component == 4) {
        textureDesc.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    } else {
        SDL_Log("Unknown component count: %d", component);
        return;
    }
    texture = std::unique_ptr<SDL_GPUTexture, std::function<void(SDL_GPUTexture*)>>(
        SDL_CreateGPUTexture(device, &textureDesc),
        [device](SDL_GPUTexture* tex) { SDL_ReleaseGPUTexture(device, tex); }
    );
}

void Image::Stage(SDL_GPUDevice* device, SDL_GPUTransferBuffer* transferBuffer, Uint32 offset) {
    void* transferData = SDL_MapGPUTransferBuffer(device, transferBuffer, false);
    memcpy(reinterpret_cast<Uint8*>(transferData) + offset, pixels.data(), pixels.size());
    SDL_UnmapGPUTransferBuffer(device, transferBuffer);
}

void Image::Upload(SDL_GPUDevice* device, SDL_GPUCopyPass* copyPass, SDL_GPUTransferBuffer* transferBuffer, Uint32 offset) {
    SDL_GPUTextureTransferInfo src = {
        .transfer_buffer = transferBuffer,
        .offset = offset
    };
    SDL_GPUTextureRegion dst = {
        .texture = texture.get(),
        .layer = 0,
        .x = 0,
        .y = 0,
        .w = static_cast<Uint32>(width),
        .h = static_cast<Uint32>(height),
        .d = 1
    };
    SDL_UploadToGPUTexture(copyPass, &src, &dst, false);
}

void Image::Release(SDL_GPUDevice* device) {
}

void Image::GenerateMipmaps(SDL_GPUCommandBuffer* cmd) {
    if (texture == nullptr || num_levels <= 1) {
        return;
    }
    SDL_GenerateMipmapsForGPUTexture(cmd, texture.get());
}