#pragma once
#include <SDL3/SDL.h>
#include "scene.hpp"
#include <memory>

SDL_GPUShader* LoadShader(
	SDL_GPUDevice* device,
	const char* filename,
	Uint32 samplerCount,
	Uint32 uniformBufferCount,
	Uint32 storageBufferCount,
	Uint32 storageTextureCount
);

SDL_GPUComputePipeline* CreateComputePipelineFromShader(
    SDL_GPUDevice* device,
    const char* filename,
    SDL_GPUComputePipelineCreateInfo* createInfo
);

const char* LoadFile(const char* filename);

std::shared_ptr<SDL_Surface> LoadImage(const char* filename);

std::shared_ptr<Scene> LoadGLTF(
    SDL_GPUDevice* device,
    const char* filename
);

void Unload(
    SDL_GPUDevice* device,
    std::shared_ptr<Scene> scene
);