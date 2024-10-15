#include "SDL3/SDL.h"
#define SDL_GPU_SHADERCROSS_IMPLEMENTATION
#include "SDL_gpu_shadercross.h"
#include <unordered_map>
#include <array>

#include "helper.hpp"

#include "fmt/core.h"


bool UseWireframeMode = false;
bool UseSmallViewport = false;
bool UseScissorRect = false;

SDL_GPUViewport SmallViewport = { 160, 120, 320, 240, 0.1f, 1.0f };
SDL_Rect ScissorRect = { 320, 240, 320, 240 };

struct PositionTextureVertex {
    float x, y, z;
    float u, v;
};
std::array<PositionTextureVertex, 6> quad = {{
    { -1, -1, 0, 0, 0 },
    {  1, -1, 0, 1, 0 },
	{  1,  1, 0, 1, 1 },
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

int windowWidth, windowHeight;

int main(int argc, char* args[]) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL could not initialize! Error: %s\n", SDL_GetError());
        return -1;
    }
    SDL_GPUDevice* device = SDL_CreateGPUDevice(SDL_ShaderCross_GetSPIRVShaderFormats(), true, NULL);
	if (device == NULL) {
		SDL_Log("GPUCreateDevice failed");
		return -1;
	}
    SDL_Window* window = SDL_CreateWindow("SDL3 GPU Demo", 500, 500, SDL_WINDOW_RESIZABLE);
	if (!SDL_ClaimWindowForGPUDevice(device, window)) {
		fmt::print("GPUClaimWindow failed");
		return -1;
	}

    SDL_GetWindowSizeInPixels(window, &windowWidth, &windowHeight);

    // Create shaders
	SDL_GPUShader* vertexShader = LoadShader(device, "TexturedQuad.vert", 0, 0, 0, 0);
	if (vertexShader == NULL) {
		SDL_Log("Failed to create vertex shader!");
		return -1;
	}

	SDL_GPUShader* fragmentShader = LoadShader(device, "TexturedQuad.frag", 1, 0, 0, 0);
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
    SDL_GPUTextureCreateInfo textureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = static_cast<uint32_t>(windowWidth),
        .height = static_cast<uint32_t>(windowHeight),
        .layer_count_or_depth = 1,
        .num_levels = 1,
    };
    SDL_GPUTexture* procTexture = SDL_CreateGPUTexture(device, &textureCreateInfo);

    SDL_GPUSamplerCreateInfo samplerCreateInfo = {
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT
    };
    SDL_GPUSampler* sampler = SDL_CreateGPUSampler(device, &samplerCreateInfo);

    // Create buffers
    SDL_GPUBufferCreateInfo bufferCreateInfo = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = sizeof(PositionTextureVertex) * quad.size()
    };
    SDL_GPUBuffer* vertexBuffer = SDL_CreateGPUBuffer(device, &bufferCreateInfo);

    SDL_GPUTransferBufferCreateInfo transferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = sizeof(PositionTextureVertex) * quad.size()
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
    memcpy(transferData, quad.data(), sizeof(PositionTextureVertex) * quad.size());
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
        .size = sizeof(PositionTextureVertex) * quad.size()
    };
	SDL_UploadToGPUBuffer(
		copyPass,
		&transferBufferLocation,
		&bufferRegion,
		false
	);
	SDL_EndGPUCopyPass(copyPass);

    SDL_SubmitGPUCommandBuffer(cmd);

    SDL_ReleaseGPUTransferBuffer(device, transferBuffer);

    // Render loop
    // SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    // SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

    std::unordered_map<SDL_Scancode, bool> keyboardState;

    bool quit = false;
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

        // SDL_RenderClear(renderer);
        // SDL_RenderPresent(renderer);

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
        SDL_DispatchGPUCompute(computePass, ceil(windowWidth / 16.0), ceil(windowHeight / 16.0), 1);
        SDL_EndGPUComputePass(computePass);

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
            SDL_BindGPUGraphicsPipeline(renderPass, UseWireframeMode ? linePipeline : fillPipeline);
            // SDL_SetGPUViewport(renderPass, &SmallViewport);
            // SDL_SetGPUScissor(renderPass, &ScissorRect);
            SDL_GPUBufferBinding vertexBufferBinding = { .buffer = vertexBuffer, .offset = 0 };
            SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBufferBinding, 1);
            SDL_GPUTextureSamplerBinding textureSamplerBinding = { .texture = procTexture, .sampler = sampler };
            SDL_BindGPUFragmentSamplers(renderPass, 0, &textureSamplerBinding, 1);
            SDL_DrawGPUPrimitives(renderPass, quad.size(), 1, 0, 0);
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

    // Release window and GPU device
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
	SDL_DestroyGPUDevice(device);
    SDL_Quit();

    return 0;
}