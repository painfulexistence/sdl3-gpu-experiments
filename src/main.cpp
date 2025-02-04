#include "fmt/core.h"
#include "SDL3/SDL.h"
#define SDL_GPU_SHADERCROSS_IMPLEMENTATION
#include "SDL_gpu_shadercross.h"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

#include <cmath>
#include <unordered_map>
#include <array>

#include "helper.hpp"


bool useWireframeMode = false;
bool useSmallViewport = false;
bool useScissorRect = false;

SDL_GPUViewport SmallViewport = { 150, 150, 200, 200, 0.1f, 1.0f };
SDL_Rect ScissorRect = { 250, 250, 125, 125 };

struct PositionTextureVertex {
    float x, y, z;
    float u, v;
};
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

struct Transform {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

int windowWidth, windowHeight;

int main(int argc, char* args[]) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL could not initialize! Error: %s\n", SDL_GetError());
        return -1;
    }
    SDL_GPUDevice* device = SDL_CreateGPUDevice(SDL_ShaderCross_GetSPIRVShaderFormats(), true, NULL);
	if (device == NULL) {
		SDL_Log("GPUCreateDevice failed: %s", SDL_GetError());
		return -1;
	}
    SDL_Window* window = SDL_CreateWindow("SDL3 GPU Demo", 1000, 1000, SDL_WINDOW_RESIZABLE);
	if (!SDL_ClaimWindowForGPUDevice(device, window)) {
		fmt::print("GPUClaimWindow failed");
		return -1;
	}

    SDL_GetWindowSizeInPixels(window, &windowWidth, &windowHeight);

    // Create shaders
	SDL_GPUShader* vertexShader = LoadShader(device, "TexturedCube.vert", 0, 1, 0, 0);
	if (vertexShader == NULL) {
		SDL_Log("Failed to create vertex shader!");
		return -1;
	}

	SDL_GPUShader* fragmentShader = LoadShader(device, "TextureColor.frag", 1, 0, 0, 0);
	if (fragmentShader == NULL) {
		SDL_Log("Failed to create fragment shader!");
		return -1;
	}

    // Create compute pipelines
    SDL_GPUComputePipelineCreateInfo compPipelineCreateInfo = {
        .num_readwrite_storage_textures = 1,
        .num_uniform_buffers = 1,
        .threadcount_x = 16,
        .threadcount_y = 16,
        .threadcount_z = 1,
    };
    SDL_GPUComputePipeline* procTexturePipeline = CreateComputePipelineFromShader(
	    device,
	    "Mandelbrot.comp",
        &compPipelineCreateInfo
    );

    // Create gfx pipelines
    SDL_GPUTextureFormat renderTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
    SDL_GPUSampleCount msaaSampleCount = SDL_GPU_SAMPLECOUNT_4;
    if (!SDL_GPUTextureSupportsSampleCount(device, renderTargetFormat, msaaSampleCount)) {
		SDL_Log("Sample count %d is not supported", (1 << static_cast<int>(msaaSampleCount)));
        msaaSampleCount = SDL_GPU_SAMPLECOUNT_4;
	}
	SDL_GPUGraphicsPipelineCreateInfo gfxPipelineCreateInfo = {
        .vertex_shader = vertexShader,
		.fragment_shader = fragmentShader,
        .vertex_input_state = (SDL_GPUVertexInputState){
			.vertex_buffer_descriptions = (SDL_GPUVertexBufferDescription[]){{
				.slot = 0,
				.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
				.instance_step_rate = 0,
				.pitch = sizeof(PositionTextureVertex)
			}},
            .num_vertex_buffers = 1,
			.vertex_attributes = (SDL_GPUVertexAttribute[]){{
				.buffer_slot = 0,
				.format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
				.location = 0,
				.offset = 0
			}, {
				.buffer_slot = 0,
				.format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
				.location = 1,
				.offset = sizeof(float) * 3
			}},
            .num_vertex_attributes = 2,
		},
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .cull_mode = SDL_GPU_CULLMODE_BACK,
        },
        .multisample_state = {
            .sample_count = msaaSampleCount
        },
        .target_info = {
			.color_target_descriptions = (SDL_GPUColorTargetDescription[]){{
				.format = renderTargetFormat
			}},
            .num_color_targets = 1,
		},
	};
	gfxPipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
	SDL_GPUGraphicsPipeline* fillPipeline = SDL_CreateGPUGraphicsPipeline(device, &gfxPipelineCreateInfo);
	if (fillPipeline == NULL) {
		SDL_Log("Failed to create fill pipeline!");
		return -1;
	}
	gfxPipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_LINE;
	SDL_GPUGraphicsPipeline* linePipeline = SDL_CreateGPUGraphicsPipeline(device, &gfxPipelineCreateInfo);
	if (linePipeline == NULL) {
		SDL_Log("Failed to create line pipeline!");
		return -1;
	}

	SDL_ReleaseGPUShader(device, vertexShader);
	SDL_ReleaseGPUShader(device, fragmentShader);

    // Create textures & samplers
    SDL_GPUTextureCreateInfo imgTextureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = 1024,
        .height = 1024,
        .layer_count_or_depth = 1,
        .num_levels = 10,
    };
    SDL_GPUTexture* imgTexture = SDL_CreateGPUTexture(device, &imgTextureCreateInfo);

    int procTextureSize = 1024;
    SDL_GPUTextureCreateInfo procTextureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = static_cast<uint32_t>(procTextureSize),
        .height = static_cast<uint32_t>(procTextureSize),
        .layer_count_or_depth = 1,
        .num_levels = static_cast<Uint32>(std::floor(std::log2(procTextureSize))),
    };
    SDL_GPUTexture* procTexture = SDL_CreateGPUTexture(device, &procTextureCreateInfo);

    SDL_GPUSamplerCreateInfo samplerCreateInfo = {
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_MIRRORED_REPEAT,
        .max_anisotropy = 4,
        .min_lod = 0.0f,
        .max_lod = 200.0f,
        .enable_anisotropy = true,
    };
    SDL_GPUSampler* sampler = SDL_CreateGPUSampler(device, &samplerCreateInfo);

    SDL_GPUTextureCreateInfo msaaTextureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = renderTargetFormat,
        .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = static_cast<uint32_t>(windowWidth),
        .height = static_cast<uint32_t>(windowHeight),
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = msaaSampleCount,
    };
    SDL_GPUTexture* msaaTexture = SDL_CreateGPUTexture(device, &msaaTextureCreateInfo);

    SDL_GPUTextureCreateInfo resolveTextureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = renderTargetFormat,
        .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = static_cast<uint32_t>(windowWidth),
        .height = static_cast<uint32_t>(windowHeight),
        .layer_count_or_depth = 1,
        .num_levels = 1,
    };
    SDL_GPUTexture* resolveTexture = SDL_CreateGPUTexture(device, &resolveTextureCreateInfo);

    // Create buffers
    SDL_GPUBufferCreateInfo vertexBufferCreateInfo = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = sizeof(PositionTextureVertex) * cube.size()
    };
    SDL_GPUBuffer* vertexBuffer = SDL_CreateGPUBuffer(device, &vertexBufferCreateInfo);

    SDL_GPUBufferCreateInfo indexBufferCreateInfo = {
        .usage = SDL_GPU_BUFFERUSAGE_INDEX,
        .size = sizeof(Uint16) * cubeIndices.size()
    };
    SDL_GPUBuffer* indexBuffer = SDL_CreateGPUBuffer(device, &indexBufferCreateInfo);

    SDL_GPUTransferBufferCreateInfo bufTransferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = sizeof(PositionTextureVertex) * cube.size() + sizeof(Uint16) * cubeIndices.size()
    };
    SDL_GPUTransferBuffer* bufTransferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &bufTransferBufferCreateInfo
	);

	PositionTextureVertex* bufTransferData = reinterpret_cast<PositionTextureVertex*>(
        SDL_MapGPUTransferBuffer(
            device,
            bufTransferBuffer,
            false
        )
	);
    memcpy(bufTransferData, cube.data(), sizeof(PositionTextureVertex) * cube.size());
    memcpy((Uint16*)&bufTransferData[cube.size()], cubeIndices.data(), sizeof(Uint16) * cubeIndices.size());
	SDL_UnmapGPUTransferBuffer(device, bufTransferBuffer);

    SDL_Surface* img = LoadImage("res/textures/rick_roll.png");

    SDL_GPUTransferBufferCreateInfo texTransferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = static_cast<Uint32>(img->w * img->h * 4)
    };
    SDL_GPUTransferBuffer* texTransferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &texTransferBufferCreateInfo
	);

	SDL_Surface* texTransferData = reinterpret_cast<SDL_Surface*>(
        SDL_MapGPUTransferBuffer(
            device,
            texTransferBuffer,
            false
        )
	);
    memcpy(texTransferData, img->pixels, img->w * img->h * 4);
	SDL_UnmapGPUTransferBuffer(device, texTransferBuffer);

    // Upload data to GPU
	SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);

	SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation bufTransferInfo = {
        .transfer_buffer = bufTransferBuffer,
        .offset = 0
    };
    SDL_GPUBufferRegion bufTransferRegion = {
        .buffer = vertexBuffer,
        .offset = 0,
        .size = sizeof(PositionTextureVertex) * cube.size()
    };
	SDL_UploadToGPUBuffer(
		copyPass,
		&bufTransferInfo,
		&bufTransferRegion,
		false
	);
    bufTransferInfo.offset = sizeof(PositionTextureVertex) * cube.size();
    bufTransferRegion.buffer = indexBuffer;
    bufTransferRegion.size = sizeof(Uint16) * cubeIndices.size();
	SDL_UploadToGPUBuffer(
		copyPass,
		&bufTransferInfo,
		&bufTransferRegion,
		false
	);
    SDL_GPUTextureTransferInfo texTransferInfo = {
        .transfer_buffer = texTransferBuffer,
        .offset = 0
    };
    SDL_GPUTextureRegion texTransferRegion = {
        .texture = imgTexture,
        .layer = 0,
        .w = static_cast<Uint32>(img->w),
        .h = static_cast<Uint32>(img->h),
        .d = 1
    };
	SDL_UploadToGPUTexture(
		copyPass,
		&texTransferInfo,
		&texTransferRegion,
		false
	);
	SDL_EndGPUCopyPass(copyPass);

    SDL_GenerateMipmapsForGPUTexture(cmd, imgTexture);

    SDL_SubmitGPUCommandBuffer(cmd);

    SDL_ReleaseGPUTransferBuffer(device, bufTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, texTransferBuffer);
    SDL_DestroySurface(img);

    // Main loop
    bool quit = false;
    std::unordered_map<SDL_Scancode, bool> keyboardState;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e) != 0) {
            switch (e.type) {
            case SDL_EVENT_QUIT:
                quit = true;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (e.key.scancode == SDL_SCANCODE_ESCAPE) {
                    quit = true;
                }
                keyboardState[e.key.scancode] = true;
                break;
            case SDL_EVENT_KEY_UP:
                keyboardState[e.key.scancode] = false;
                break;
            case SDL_EVENT_WINDOW_RESIZED:

                break;
            default:
                break;
            }
        }

        if (keyboardState[SDL_SCANCODE_W]) {
            useWireframeMode = !useWireframeMode;
        }
        if (keyboardState[SDL_SCANCODE_V]) {
            useSmallViewport = !useSmallViewport;
        }
        if (keyboardState[SDL_SCANCODE_S]) {
            useScissorRect = !useScissorRect;
        }

        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        if (cmd == NULL) {
            SDL_Log("AcquireGPUCommandBuffer failed: %s", SDL_GetError());
            return -1;
        }

        // 1. compute pass
        SDL_GPUComputePass* computePass = SDL_BeginGPUComputePass(
            cmd,
            (SDL_GPUStorageTextureReadWriteBinding[]){{
                .texture = procTexture,
                .mip_level = 0,
            }},
            1,
            NULL,
            0
        );
        SDL_BindGPUComputePipeline(computePass, procTexturePipeline);
        float time = SDL_GetTicks() / 1000.0;
        SDL_PushGPUComputeUniformData(cmd, 0, &time, sizeof(float));
        SDL_DispatchGPUCompute(computePass, ceil(procTextureSize / 16.0), ceil(procTextureSize / 16.0), 1);
        SDL_EndGPUComputePass(computePass);

        SDL_GenerateMipmapsForGPUTexture(cmd, procTexture);

        // 2. screen pass
        SDL_GPUTexture* swapchainTexture;
        if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, window, &swapchainTexture, NULL, NULL)) {
            SDL_Log("WaitAndAcquireGPUSwapchainTexture failed: %s", SDL_GetError());
            return -1;
        }
        if (swapchainTexture != NULL) {
            SDL_GPUColorTargetInfo colorTargetInfo = { 0 };
            colorTargetInfo.texture = msaaTexture;
            colorTargetInfo.clear_color = (SDL_FColor){ 0.0f, 0.0f, 0.0f, 1.0f };
            colorTargetInfo.load_op = SDL_GPU_LOADOP_CLEAR;
            colorTargetInfo.store_op = SDL_GPU_STOREOP_RESOLVE;
            colorTargetInfo.resolve_texture = resolveTexture;

            SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(cmd, &colorTargetInfo, 1, NULL);
            SDL_BindGPUGraphicsPipeline(renderPass, useWireframeMode ? linePipeline : fillPipeline);
            if (useSmallViewport) SDL_SetGPUViewport(renderPass, &SmallViewport);
            if (useScissorRect) SDL_SetGPUScissor(renderPass, &ScissorRect);
            SDL_GPUBufferBinding vertexBufferBinding = { .buffer = vertexBuffer, .offset = 0 };
            SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBufferBinding, 1);
            SDL_GPUBufferBinding indexBufferBinding = { .buffer = indexBuffer, .offset = 0 };
            SDL_BindGPUIndexBuffer(renderPass, &indexBufferBinding, SDL_GPU_INDEXELEMENTSIZE_16BIT);
            SDL_GPUTextureSamplerBinding textureSamplerBinding = { .texture = procTexture, .sampler = sampler };
            SDL_BindGPUFragmentSamplers(renderPass, 0, &textureSamplerBinding, 1);
            float angle = time * 0.5f;
            Transform xform;
            xform.model = glm::rotate(glm::identity<glm::mat4>(), angle, glm::vec3(0.0f, 1.0f, -1.0f));
            glm::vec3  camPos = glm::vec3(0.0f, 0.0f, 3.0f);
            xform.proj = glm::perspective(45.f * (float)M_PI / 180.f, windowWidth / (float)windowHeight, 0.03f, 500.0f);
            xform.view = glm::lookAt(camPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
            SDL_PushGPUVertexUniformData(cmd, 0, &xform, sizeof(Transform));
            // SDL_DrawGPUPrimitives(renderPass, quad.size(), 1, 0, 0);
            SDL_DrawGPUIndexedPrimitives(renderPass, cubeIndices.size(), 1, 0, 0, 0);
            SDL_EndGPURenderPass(renderPass);

            SDL_GPUBlitInfo blitInfo = {
				.source.texture = resolveTexture,
				.source.w = static_cast<Uint32>(windowWidth),
				.source.h = static_cast<Uint32>(windowHeight),
				.destination.texture = swapchainTexture,
				.destination.w = static_cast<Uint32>(windowWidth),
				.destination.h = static_cast<Uint32>(windowHeight),
				.load_op = SDL_GPU_LOADOP_DONT_CARE,
				.filter = SDL_GPU_FILTER_LINEAR
			};
            SDL_BlitGPUTexture(cmd, &blitInfo);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
    }

    // Release GPU resources
    SDL_ReleaseGPUComputePipeline(device, procTexturePipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, fillPipeline);
	SDL_ReleaseGPUGraphicsPipeline(device, linePipeline);
    SDL_ReleaseGPUTexture(device, imgTexture);
    SDL_ReleaseGPUTexture(device, procTexture);
    SDL_ReleaseGPUSampler(device, sampler);
    SDL_ReleaseGPUTexture(device, msaaTexture);
    SDL_ReleaseGPUTexture(device, resolveTexture);
    SDL_ReleaseGPUBuffer(device, vertexBuffer);
    SDL_ReleaseGPUBuffer(device, indexBuffer);

    // Release window and GPU device
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
	SDL_DestroyGPUDevice(device);
    SDL_Quit();

    return 0;
}