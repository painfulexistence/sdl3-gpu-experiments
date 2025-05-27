#pragma once
#include <SDL3/SDL.h>
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

std::shared_ptr<SDL_Surface> LoadImage(const char* filename);

std::shared_ptr<Scene> LoadGLTF(
    SDL_GPUDevice* device,
    const char* filename
);