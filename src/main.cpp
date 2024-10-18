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
    SDL_Window* window = SDL_CreateWindow("SDL3 GPU Demo", 500, 500, SDL_WINDOW_RESIZABLE);
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
        .target_info = {
			.color_target_descriptions = (SDL_GPUColorTargetDescription[]){{
				.format = SDL_GetGPUSwapchainTextureFormat(device, window)
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
    int procTextureSize = 1024;
    SDL_GPUTextureCreateInfo procTextureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = static_cast<uint32_t>(procTextureSize),
        .height = static_cast<uint32_t>(procTextureSize),
        .layer_count_or_depth = 1,
        .num_levels = static_cast <Uint32>(std::floor(std::log(procTextureSize))),
    };
    SDL_GPUTexture* procTexture = SDL_CreateGPUTexture(device, &procTextureCreateInfo);

    SDL_GPUSamplerCreateInfo samplerCreateInfo = {
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT
    };
    SDL_GPUSampler* sampler = SDL_CreateGPUSampler(device, &samplerCreateInfo);

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

    SDL_GPUTransferBufferCreateInfo transferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = sizeof(PositionTextureVertex) * cube.size() + sizeof(Uint16) * cubeIndices.size()
    };
    SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &transferBufferCreateInfo
	);

	PositionTextureVertex* transferData = reinterpret_cast<PositionTextureVertex*>(
        SDL_MapGPUTransferBuffer(
            device,
            transferBuffer,
            false
        )
	);
    memcpy(transferData, cube.data(), sizeof(PositionTextureVertex) * cube.size());
    memcpy((Uint16*)&transferData[cube.size()], cubeIndices.data(), sizeof(Uint16) * cubeIndices.size());
	SDL_UnmapGPUTransferBuffer(device, transferBuffer);

    // Upload data to GPU
	SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);

	SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation transferBufferLocation = {
        .transfer_buffer = transferBuffer,
        .offset = 0
    };
    SDL_GPUBufferRegion bufferRegion = {
        .buffer = vertexBuffer,
        .offset = 0,
        .size = sizeof(PositionTextureVertex) * cube.size()
    };
	SDL_UploadToGPUBuffer(
		copyPass,
		&transferBufferLocation,
		&bufferRegion,
		false
	);
    transferBufferLocation.offset = sizeof(PositionTextureVertex) * cube.size();
    bufferRegion.buffer = indexBuffer;
    bufferRegion.size = sizeof(Uint16) * cubeIndices.size();
	SDL_UploadToGPUBuffer(
		copyPass,
		&transferBufferLocation,
		&bufferRegion,
		false
	);
	SDL_EndGPUCopyPass(copyPass);

    SDL_SubmitGPUCommandBuffer(cmd);

    SDL_ReleaseGPUTransferBuffer(device, transferBuffer);

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
        if (!SDL_AcquireGPUSwapchainTexture(cmd, window, &swapchainTexture, NULL, NULL)) {
            SDL_Log("AcquireGPUSwapchainTexture failed: %s", SDL_GetError());
            return -1;
        }
        if (swapchainTexture != NULL) {
            SDL_GPUColorTargetInfo colorTargetInfo = { 0 };
            colorTargetInfo.texture = swapchainTexture;
            colorTargetInfo.clear_color = (SDL_FColor){ 0.0f, 0.0f, 0.0f, 1.0f };
            colorTargetInfo.load_op = SDL_GPU_LOADOP_CLEAR;
            colorTargetInfo.store_op = SDL_GPU_STOREOP_STORE;

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
        }

        SDL_SubmitGPUCommandBuffer(cmd);
    }

    // Release GPU resources
    SDL_ReleaseGPUComputePipeline(device, procTexturePipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, fillPipeline);
	SDL_ReleaseGPUGraphicsPipeline(device, linePipeline);
    SDL_ReleaseGPUTexture(device, procTexture);
    SDL_ReleaseGPUSampler(device, sampler);
    SDL_ReleaseGPUBuffer(device, vertexBuffer);
    SDL_ReleaseGPUBuffer(device, indexBuffer);

    // Release window and GPU device
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
	SDL_DestroyGPUDevice(device);
    SDL_Quit();

    return 0;
}