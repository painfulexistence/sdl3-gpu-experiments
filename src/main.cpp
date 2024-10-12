#include "SDL3/SDL.h"
#define SDL_GPU_SHADERCROSS_IMPLEMENTATION
#include "SDL_gpu_shadercross.h"

#include "helper.hpp"

#include "fmt/core.h"

bool UseWireframeMode = false;
bool UseSmallViewport = false;
bool UseScissorRect = false;

SDL_GPUViewport SmallViewport = { 160, 120, 320, 240, 0.1f, 1.0f };
SDL_Rect ScissorRect = { 320, 240, 320, 240 };

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
    SDL_Window* window = SDL_CreateWindow("MyApp", 800, 600, SDL_WINDOW_RESIZABLE);
	if (!SDL_ClaimWindowForGPUDevice(device, window)) {
		fmt::print("GPUClaimWindow failed");
		return -1;
	}

    // Create shaders
	SDL_GPUShader* vertexShader = LoadShader(device, "Triforce.vert", 0, 0, 0, 0);
	if (vertexShader == NULL) {
		SDL_Log("Failed to create vertex shader!");
		return -1;
	}

	SDL_GPUShader* fragmentShader = LoadShader(device, "SolidColor.frag", 0, 0, 0, 0);
	if (fragmentShader == NULL) {
		SDL_Log("Failed to create fragment shader!");
		return -1;
	}

    // Create gfx pipelines
	SDL_GPUGraphicsPipelineCreateInfo pipelineCreateInfo = {
		.vertex_shader = vertexShader,
		.fragment_shader = fragmentShader,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .target_info = {
			.color_target_descriptions = (SDL_GPUColorTargetDescription[]){{
				.format = SDL_GetGPUSwapchainTextureFormat(device, window)
			}},
            .num_color_targets = 1,
		},
	};

	pipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
	SDL_GPUGraphicsPipeline* FillPipeline = SDL_CreateGPUGraphicsPipeline(device, &pipelineCreateInfo);
	if (FillPipeline == NULL) {
		SDL_Log("Failed to create fill pipeline!");
		return -1;
	}

	pipelineCreateInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_LINE;
	SDL_GPUGraphicsPipeline* LinePipeline = SDL_CreateGPUGraphicsPipeline(device, &pipelineCreateInfo);
	if (LinePipeline == NULL) {
		SDL_Log("Failed to create line pipeline!");
		return -1;
	}

	SDL_ReleaseGPUShader(device, vertexShader);
	SDL_ReleaseGPUShader(device, fragmentShader);

    // SDL_Renderer* renderer = SDL_CreateRenderer(window, NULL);
    // SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

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
            SDL_BindGPUGraphicsPipeline(renderPass, UseWireframeMode ? LinePipeline : FillPipeline);
            if (UseSmallViewport) {
                SDL_SetGPUViewport(renderPass, &SmallViewport);
            }
            if (UseScissorRect) {
                SDL_SetGPUScissor(renderPass, &ScissorRect);
            }
            SDL_DrawGPUPrimitives(renderPass, 12, 1, 0, 0);
            SDL_EndGPURenderPass(renderPass);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
    }
    SDL_ReleaseGPUGraphicsPipeline(device, FillPipeline);
	SDL_ReleaseGPUGraphicsPipeline(device, LinePipeline);

    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
	SDL_DestroyGPUDevice(device);
    SDL_Quit();

    return 0;
}