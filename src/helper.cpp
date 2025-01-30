#include "helper.hpp"

#include "SDL_gpu_shadercross.h"
#define STBI_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

SDL_GPUComputePipeline* CreateComputePipelineFromShader(
    SDL_GPUDevice* device,
    const char* filename,
    SDL_GPUComputePipelineCreateInfo* createInfo
) {
    const char* basePath = SDL_GetBasePath();
	char fullPath[256];
	SDL_snprintf(fullPath, sizeof(fullPath), "%sres/shaders/%s.spv", basePath, filename);

	size_t codeSize;
	void* code = SDL_LoadFile(fullPath, &codeSize);
	if (code == NULL) {
		SDL_Log("Failed to load compute shader from disk! %s", fullPath);
		return NULL;
	}

	SDL_GPUComputePipelineCreateInfo newCreateInfo = *createInfo;
	newCreateInfo.code = static_cast<const Uint8 *>(code);
	newCreateInfo.code_size = codeSize;
	newCreateInfo.entrypoint = "main";
	newCreateInfo.format = SDL_GPU_SHADERFORMAT_SPIRV;

	SDL_GPUComputePipeline* pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(device, &newCreateInfo);
	if (pipeline == NULL) {
		SDL_Log("Failed to create compute pipeline!");
		SDL_free(code);
		return NULL;
	}

	SDL_free(code);
	return pipeline;
}

SDL_GPUShader* LoadShader(
	SDL_GPUDevice* device,
	const char* filename,
	Uint32 samplerCount,
	Uint32 uniformBufferCount,
	Uint32 storageBufferCount,
	Uint32 storageTextureCount
) {
	SDL_GPUShaderStage stage;
	if (SDL_strstr(filename, ".vert")) {
		stage = SDL_GPU_SHADERSTAGE_VERTEX;
	} else if (SDL_strstr(filename, ".frag")) {
		stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
	} else {
		SDL_Log("Invalid shader stage!");
		return NULL;
	}

    const char* basePath = SDL_GetBasePath();
	char fullPath[256];
	SDL_snprintf(fullPath, sizeof(fullPath), "%sres/shaders/%s.spv", basePath, filename);

	size_t codeSize;
	void* code = SDL_LoadFile(fullPath, &codeSize);
	if (code == NULL) {
		SDL_Log("Failed to load shader from disk! %s", fullPath);
		return NULL;
	}

	SDL_GPUShaderCreateInfo shaderInfo = {
		.code_size = codeSize,
        .code = static_cast<const Uint8 *>(code),
		.entrypoint = "main",
		.format = SDL_GPU_SHADERFORMAT_SPIRV,
		.stage = stage,
		.num_samplers = samplerCount,
        .num_storage_textures = storageTextureCount,
		.num_storage_buffers = storageBufferCount,
        .num_uniform_buffers = uniformBufferCount,
	};
	SDL_GPUShader* shader = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(device, &shaderInfo);
	if (shader == NULL) {
		SDL_Log("Failed to create shader!");
		SDL_free(code);
		return NULL;
	}

	SDL_free(code);
	return shader;
}

SDL_Surface* LoadImage(const char* filename) {
    int width, height, numChannels;
    if (!stbi_info(filename, &width, &height, &numChannels)) {
        SDL_Log("Failed to load image at %s!\n", filename);
        return nullptr;
    }
    SDL_PixelFormat desiredFormat;
    switch (numChannels) {
    case 3:
    case 4:
        desiredFormat = SDL_PixelFormat::SDL_PIXELFORMAT_RGBA32;
        break;
    default:
        SDL_Log("Unknown texture format at %s\n", filename);
        return nullptr;
    }
    uint8_t* data = stbi_load(filename, &width, &height, &numChannels, 4);
    if (data) {
        auto image = SDL_CreateSurfaceFrom(width, height, desiredFormat, data, width * 32);
        stbi_image_free(data);
        return image;
    } else {
        return nullptr;
    }
}