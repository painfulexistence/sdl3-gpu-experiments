#pragma once
#include <SDL3/SDL.h>
#include "image.hpp"
#include "scene.hpp"
#include <memory>

SDL_GPUShader* LoadShader(
	SDL_GPUDevice* device,
	const char* filename,
	Uint32 samplerCount = 0,
    Uint32 uniformBufferCount = 0,
	Uint32 storageBufferCount = 0,
	Uint32 storageTextureCount = 0
);

SDL_GPUComputePipeline* CreateComputePipelineFromShader(
    SDL_GPUDevice* device,
    const char* filename,
    Uint32 samplerCount = 0,
    Uint32 uniformBufferCount = 0,
    Uint32 readonlyStorageBufferCount = 0,
    Uint32 readonlyStorageTextureCount = 0,
	Uint32 readwriteStorageBufferCount = 0,
    Uint32 readwriteStorageTextureCount = 0,
    Uint32 threadCountX = 1,
    Uint32 threadCountY = 1,
    Uint32 threadCountZ = 1
);

std::shared_ptr<Image> LoadImage(const char* filename);

// Loads an equirectangular HDR environment map (.hdr) into an
// R32G32B32A32_FLOAT sampler texture, uploading it via its own copy pass.
// Returns nullptr if the file is missing/unreadable (the skybox is then skipped).
SDL_GPUTexture* CreateHDRTexture(SDL_GPUDevice* device, const char* filename);

std::shared_ptr<Scene> LoadGLTF(
    SDL_GPUDevice* device,
    const char* filename
);