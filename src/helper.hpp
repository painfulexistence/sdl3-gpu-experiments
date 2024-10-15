#pragma once
#include <SDL3/SDL.h>

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