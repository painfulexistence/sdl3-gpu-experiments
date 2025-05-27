#include "fmt/core.h"
#include "SDL3/SDL.h"
#define SDL_GPU_SHADERCROSS_IMPLEMENTATION
#include "SDL_gpu_shadercross.h"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "glm/mat4x4.hpp"
#include "glm/ext/matrix_transform.hpp"

#include <cmath>
#include <unordered_map>
#include <array>
#include <functional>

#include "helper.hpp"
#include "camera.hpp"
#include "geometry.hpp"

bool useWireframeMode = false;
bool useSmallViewport = false;
bool useScissorRect = false;

SDL_GPUViewport SmallViewport = { 150, 150, 200, 200, 0.1f, 1.0f };
SDL_Rect ScissorRect = { 250, 250, 125, 125 };


struct CameraInfo {
    glm::mat4 view;
    glm::mat4 proj;
};

struct SimpleInstance {
    glm::mat4 model;
};

struct Instance {
    glm::mat4 model;
    Uint32 meshID;
    Uint32 materialID;
};

struct MeshInfo {
    Uint32 baseVertex;
    Uint32 baseIndex;
    Uint32 indexCount;
};

struct MaterialInfo {
    // Uint8 albedoMap;
    // Uint8 normalMap;
    // Uint8 metallicRoughnessMap;
    // Uint8 aoMap;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float roughness;
};

struct DrawCommand {
    Uint32 indexCount;
    Uint32 instanceCount;
    Uint32 firstIndex;
    Uint32 baseVertex;
    Uint32 baseInstance;
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
	SDL_Log("Create shaders");
	SDL_GPUShader* simpleVertexShader = LoadShader(device, "TexturedCube.vert", 0, 2, 0, 0);
	if (simpleVertexShader == NULL) {
		SDL_Log("Failed to create vertex shader!");
		return -1;
	}

	SDL_GPUShader* simpleFragmentShader = LoadShader(device, "TextureColor.frag", 1, 0, 0, 0);
	if (simpleFragmentShader == NULL) {
		SDL_Log("Failed to create fragment shader!");
		return -1;
	}

	SDL_GPUShader* uberVertexShader = LoadShader(device, "Uber.vert", 0, 1, 5, 0);
	if (uberVertexShader == NULL) {
		SDL_Log("Failed to create vertex shader!");
		return -1;
	}

	SDL_GPUShader* uberFragmentShader = LoadShader(device, "Uber.frag", 0, 1, 1, 0);
	if (uberFragmentShader == NULL) {
		SDL_Log("Failed to create fragment shader!");
		return -1;
	}

    // Create compute pipelines
    SDL_Log("Create compute pipelines");
    SDL_GPUComputePipeline* procTexturePipeline = CreateComputePipelineFromShader(
	    device,
	    "Mandelbrot.comp",
        0, 1, 0, 0, 0, 1, 16, 16, 1
    );

    SDL_GPUComputePipeline* cullingPipeline = CreateComputePipelineFromShader(
	    device,
	    "Cull.comp",
        0, 1, 1, 0, 2, 0, 64, 1, 1
    );

    SDL_GPUComputePipeline* commandBuildingPipeline = CreateComputePipelineFromShader(
	    device,
	    "CommandBuild.comp",
        0, 0, 3, 0, 1, 0, 64, 1, 1
    );

    SDL_GPUComputePipeline* prefixSumPipeline = CreateComputePipelineFromShader(
	    device,
	    "SerializedPrefixSum.comp",
        0, 0, 0, 0, 2, 0, 1, 1, 1
    );

    SDL_GPUComputePipeline* resetCounterPipeline = CreateComputePipelineFromShader(
        device,
        "ResetCounter.comp",
        0, 0, 0, 0, 1, 0, 1, 1, 1
    );

    // Create gfx pipelines
    SDL_Log("Create gfx pipelines");
    SDL_GPUTextureFormat renderTargetFormat = SDL_GetGPUSwapchainTextureFormat(device, window);
    SDL_GPUSampleCount msaaSampleCount = SDL_GPU_SAMPLECOUNT_4;
    if (!SDL_GPUTextureSupportsSampleCount(device, renderTargetFormat, msaaSampleCount)) {
		SDL_Log("Sample count %d is not supported", (1 << static_cast<int>(msaaSampleCount)));
        msaaSampleCount = SDL_GPU_SAMPLECOUNT_4;
	}

    std::array<SDL_GPUVertexBufferDescription, 1> simpleVertexBufferDescs = {{
        {
            .slot = 0,
            .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
            .instance_step_rate = 0,
            .pitch = sizeof(PositionTextureVertex)
        }
    }};
    std::array<SDL_GPUVertexAttribute, 2> simpleVertexAttributes = {{
        {
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .location = 0,
            .offset = 0
        },
        {
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
            .location = 1,
            .offset = sizeof(float) * 3
        }
    }};
    std::array<SDL_GPUVertexBufferDescription, 1> vertexBufferDescs = {{
        {
            .slot = 0,
            .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
            .instance_step_rate = 0,
            .pitch = sizeof(Vertex)
        }
    }};
    std::array<SDL_GPUVertexAttribute, 3> vertexAttributes = {{
        {
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .location = 0,
            .offset = 0
        },
        {
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
            .location = 1,
            .offset = sizeof(float) * 3
        },
        {
            .buffer_slot = 0,
            .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
            .location = 2,
            .offset = sizeof(float) * 6
        }
    }};
	SDL_GPUGraphicsPipelineCreateInfo gfxPipelineDesc = {
        .vertex_shader = simpleVertexShader,
		.fragment_shader = simpleFragmentShader,
        .vertex_input_state = (SDL_GPUVertexInputState){
			.vertex_buffer_descriptions = simpleVertexBufferDescs.data(),
            .num_vertex_buffers = static_cast<Uint32>(simpleVertexBufferDescs.size()),
			.vertex_attributes = simpleVertexAttributes.data(),
            .num_vertex_attributes = static_cast<Uint32>(simpleVertexAttributes.size()),
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
	gfxPipelineDesc.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
	SDL_GPUGraphicsPipeline* fillPipeline = SDL_CreateGPUGraphicsPipeline(device, &gfxPipelineDesc);
	if (fillPipeline == NULL) {
		SDL_Log("Failed to create textured fill pipeline!");
		return -1;
	}
	gfxPipelineDesc.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_LINE;
	SDL_GPUGraphicsPipeline* linePipeline = SDL_CreateGPUGraphicsPipeline(device, &gfxPipelineDesc);
	if (linePipeline == NULL) {
		SDL_Log("Failed to create textured line pipeline!");
		return -1;
	}
    gfxPipelineDesc.vertex_shader = uberVertexShader;
    gfxPipelineDesc.fragment_shader = uberFragmentShader;
    gfxPipelineDesc.vertex_input_state = (SDL_GPUVertexInputState){
		.vertex_buffer_descriptions = vertexBufferDescs.data(),
        .num_vertex_buffers = static_cast<Uint32>(vertexBufferDescs.size()),
		.vertex_attributes = vertexAttributes.data(),
        .num_vertex_attributes = static_cast<Uint32>(vertexAttributes.size()),
	};
    SDL_GPUGraphicsPipeline* uberPipeline = SDL_CreateGPUGraphicsPipeline(device, &gfxPipelineDesc);
    if (uberPipeline == NULL) {
        SDL_Log("Failed to create uber pipeline!");
        return -1;
    }
	SDL_ReleaseGPUShader(device, simpleVertexShader);
	SDL_ReleaseGPUShader(device, simpleFragmentShader);
	SDL_ReleaseGPUShader(device, uberVertexShader);
	SDL_ReleaseGPUShader(device, uberFragmentShader);

    // Create textures & samplers
    std::shared_ptr<SDL_Surface> img = LoadImage("res/textures/rick_roll.png");
    SDL_GPUTextureCreateInfo imgTextureCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = static_cast<Uint32>(img->w),
        .height = static_cast<Uint32>(img->h),
        .layer_count_or_depth = 1,
        .num_levels = std::max(1u, static_cast<Uint32>(std::floor(std::log2(std::min(img->w, img->h))))),
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

    // Load scene
    std::shared_ptr<Scene> sponza = LoadGLTF(device, "res/models/Sponza/Sponza.gltf");
    // sponza->Print();
    auto testMesh = CPUMesh::CreateCube();
    auto testMeshInstances = std::vector<SimpleInstance>({
        {
            .model = glm::identity<glm::mat4>()
        }
    });

    // Data for SSBOs
    // FIXME: setting instance count to 1000000 is causing seg fault
    constexpr int numInstances = 200;
    constexpr int numMeshes = 10;
    constexpr int numMaterials = 100;
    constexpr int numDrawCommands = numMeshes;
    auto cubeVertices = CreateCubeVertices();
    auto cubeIndices = CreateCubeIndices();
    auto sphereVertices = CreateSphereVertices();
    auto sphereIndices = CreateSphereIndices();
    std::array<Instance, numInstances> instances = {};
    std::array<MeshInfo, numMeshes> meshInfos = {};
    std::array<MaterialInfo, numMaterials> materialInfos = {};
    std::array<Uint32, numInstances> visibleInstanceIndices = {};
    Uint32 visibleCounter = 0;
    std::array<DrawCommand, numDrawCommands> drawCommands = {};
    std::array<Uint32, numDrawCommands> prefixSums = {};

    for (int i = 0; i < instances.size(); i++) {
        instances[i] = {
            .model = glm::identity<glm::mat4>(),
            .meshID = 0,
            .materialID = 0
        };
    }

    Uint32 vertexBufferSize = sizeof(Vertex) * cubeVertices.size() + sizeof(Vertex) * sphereVertices.size();
    Uint32 indexBufferSize = sizeof(Uint32) * cubeIndices.size() + sizeof(Uint32) * sphereIndices.size();
    Uint32 instanceBufferSize = sizeof(Instance) * instances.size();
    Uint32 meshBufferSize = sizeof(MeshInfo) * meshInfos.size();
    Uint32 materialBufferSize = sizeof(MaterialInfo) * materialInfos.size();
    Uint32 visibilityBufferSize = sizeof(Uint32) * visibleInstanceIndices.size();
    Uint32 visibleCounterBufferSize = sizeof(Uint32);
    Uint32 drawCommandBufferSize = sizeof(DrawCommand) * drawCommands.size();
    Uint32 prefixSumBufferSize = sizeof(Uint32) * prefixSums.size();

    Uint32 vertexBufferOffset = 0;
    Uint32 indexBufferOffset = vertexBufferOffset + vertexBufferSize;
    Uint32 instanceBufferOffset = indexBufferOffset + indexBufferSize;
    Uint32 meshBufferOffset = instanceBufferOffset + instanceBufferSize;
    Uint32 materialBufferOffset = meshBufferOffset + meshBufferSize;
    Uint32 visibilityBufferOffset = materialBufferOffset + materialBufferSize;
    Uint32 visibleCounterBufferOffset = visibilityBufferOffset + visibilityBufferSize;
    Uint32 drawCommandBufferOffset = visibleCounterBufferOffset + visibleCounterBufferSize;
    Uint32 prefixSumBufferOffset = drawCommandBufferOffset + drawCommandBufferSize;

    Uint32 totalBufferSize = prefixSumBufferOffset + prefixSumBufferSize;

    // Create render targets
    SDL_Log("Create render targets");
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

    // Create SSBOs
    SDL_Log("Create SSBOs");
    SDL_GPUBufferCreateInfo vertexBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = vertexBufferSize
    };
    SDL_GPUBuffer* vertexBuffer = SDL_CreateGPUBuffer(device, &vertexBufferDesc);

    SDL_GPUBufferCreateInfo indexBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = indexBufferSize
    };
    SDL_GPUBuffer* indexBuffer = SDL_CreateGPUBuffer(device, &indexBufferDesc);

    SDL_GPUBufferCreateInfo instanceBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = instanceBufferSize
    };
    SDL_GPUBuffer* instanceBuffer = SDL_CreateGPUBuffer(device, &instanceBufferDesc);

    SDL_GPUBufferCreateInfo meshBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = meshBufferSize
    };
    SDL_GPUBuffer* meshBuffer = SDL_CreateGPUBuffer(device, &meshBufferDesc);

    SDL_GPUBufferCreateInfo materialBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = materialBufferSize
    };
    SDL_GPUBuffer* materialBuffer = SDL_CreateGPUBuffer(device, &materialBufferDesc);

    SDL_GPUBufferCreateInfo visibilityBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = visibilityBufferSize
    };
    SDL_GPUBuffer* visibilityBuffer = SDL_CreateGPUBuffer(device, &visibilityBufferDesc);

    SDL_GPUBufferCreateInfo visibleCounterBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = visibleCounterBufferSize
    };
    SDL_GPUBuffer* visibleCounterBuffer = SDL_CreateGPUBuffer(device, &visibleCounterBufferDesc);

    SDL_GPUBufferCreateInfo drawCommandBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_BUFFERUSAGE_INDIRECT,
        .size = drawCommandBufferSize
    };
    SDL_GPUBuffer* drawCommandBuffer = SDL_CreateGPUBuffer(device, &drawCommandBufferDesc);

    SDL_GPUBufferCreateInfo prefixSumBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = prefixSumBufferSize
    };
    SDL_GPUBuffer* prefixSumBuffer = SDL_CreateGPUBuffer(device, &prefixSumBufferDesc);

    if (vertexBuffer == NULL || indexBuffer == NULL || instanceBuffer == NULL || meshBuffer == NULL || materialBuffer == NULL || visibilityBuffer == NULL || visibleCounterBuffer == NULL || drawCommandBuffer == NULL || prefixSumBuffer == NULL) {
        SDL_Log("Failed to create SSBOs");
        return -1;
    }

    // Transfer data to staging buffer
    SDL_GPUTransferBufferCreateInfo bufTransferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = static_cast<Uint32>(testMesh.vertex_byte_count() * testMesh.vertex_count() + testMesh.index_byte_count() * testMesh.index_count())
    };
    SDL_GPUTransferBuffer* bufTransferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &bufTransferBufferCreateInfo
	);
    testMesh.Stage(device, bufTransferBuffer);

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

    SDL_GPUTransferBufferCreateInfo ssboTransferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = totalBufferSize // TODO: set to a very high value
    };
    SDL_GPUTransferBuffer* ssboTransferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &ssboTransferBufferCreateInfo
	);
    auto ssboTransferData = reinterpret_cast<Uint8*>(
        SDL_MapGPUTransferBuffer(
            device,
            ssboTransferBuffer,
            false
        )
	);
    memcpy(&ssboTransferData[vertexBufferOffset], cubeVertices.data(), sizeof(Vertex) * cubeVertices.size());
    memcpy(&ssboTransferData[vertexBufferOffset + sizeof(Vertex) * cubeVertices.size()], sphereVertices.data(), sizeof(Vertex) * sphereVertices.size());
    memcpy(&ssboTransferData[indexBufferOffset], cubeIndices.data(), sizeof(Uint32) * cubeIndices.size());
    memcpy(&ssboTransferData[indexBufferOffset + sizeof(Uint32) * cubeIndices.size()], sphereIndices.data(), sizeof(Uint32) * sphereIndices.size());
    memcpy(&ssboTransferData[instanceBufferOffset], instances.data(), instanceBufferSize);
    memcpy(&ssboTransferData[meshBufferOffset], meshInfos.data(), meshBufferSize);
    memcpy(&ssboTransferData[materialBufferOffset], materialInfos.data(), materialBufferSize);
    memcpy(&ssboTransferData[visibilityBufferOffset], visibleInstanceIndices.data(), visibilityBufferSize);
    memcpy(&ssboTransferData[visibleCounterBufferOffset], &visibleCounter, visibleCounterBufferSize);
    memcpy(&ssboTransferData[prefixSumBufferOffset], prefixSums.data(), prefixSumBufferSize);
    memcpy(&ssboTransferData[drawCommandBufferOffset], drawCommands.data(), drawCommandBufferSize);
	SDL_UnmapGPUTransferBuffer(device, ssboTransferBuffer);

    // Upload data to GPU
	SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);

	SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);

    testMesh.Upload(device, copyPass, bufTransferBuffer);

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

    SDL_GPUTransferBufferLocation ssboTransferInfo = {
        .transfer_buffer = ssboTransferBuffer,
        .offset = vertexBufferOffset
    };
    SDL_GPUBufferRegion ssboTransferRegion = {
        .buffer = vertexBuffer,
        .offset = 0,
        .size = vertexBufferSize
    };
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = indexBufferOffset;
    ssboTransferRegion.buffer = indexBuffer;
    ssboTransferRegion.size = indexBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = instanceBufferOffset;
    ssboTransferRegion.buffer = instanceBuffer;
    ssboTransferRegion.size = instanceBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = meshBufferOffset;
    ssboTransferRegion.buffer = meshBuffer;
    ssboTransferRegion.size = meshBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = materialBufferOffset;
    ssboTransferRegion.buffer = materialBuffer;
    ssboTransferRegion.size = materialBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = visibilityBufferOffset;
    ssboTransferRegion.buffer = visibilityBuffer;
    ssboTransferRegion.size = visibilityBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = visibleCounterBufferOffset;
    ssboTransferRegion.buffer = visibleCounterBuffer;
    ssboTransferRegion.size = visibleCounterBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = prefixSumBufferOffset;
    ssboTransferRegion.buffer = prefixSumBuffer;
    ssboTransferRegion.size = prefixSumBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );
    ssboTransferInfo.offset = drawCommandBufferOffset;
    ssboTransferRegion.buffer = drawCommandBuffer;
    ssboTransferRegion.size = drawCommandBufferSize;
    SDL_UploadToGPUBuffer(
        copyPass,
        &ssboTransferInfo,
        &ssboTransferRegion,
        false
    );

	SDL_EndGPUCopyPass(copyPass);

    SDL_GenerateMipmapsForGPUTexture(cmd, imgTexture);

    SDL_SubmitGPUCommandBuffer(cmd);

    SDL_ReleaseGPUTransferBuffer(device, bufTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, texTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, ssboTransferBuffer);

    Camera camera(
        glm::vec3(0.0f, 0.0f, -3.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::radians(45.0f),
        windowWidth / (float)windowHeight,
        0.1f,
        500.0f
    );

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

        if (keyboardState[SDL_SCANCODE_Z]) {
            useWireframeMode = !useWireframeMode;
        }
        if (keyboardState[SDL_SCANCODE_X]) {
            useSmallViewport = !useSmallViewport;
        }
        if (keyboardState[SDL_SCANCODE_C]) {
            useScissorRect = !useScissorRect;
        }
        if (keyboardState[SDL_SCANCODE_W]) {
            camera.Dolly(0.1f);
        }
        if (keyboardState[SDL_SCANCODE_S]) {
            camera.Dolly(-0.1f);
        }
        if (keyboardState[SDL_SCANCODE_D]) {
            camera.Truck(-0.1f);
        }
        if (keyboardState[SDL_SCANCODE_A]) {
            camera.Truck(0.1f);
        }
        if (keyboardState[SDL_SCANCODE_R]) {
            camera.Pedestal(0.1f);
        }
        if (keyboardState[SDL_SCANCODE_F]) {
            camera.Pedestal(-0.1f);
        }

        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        if (cmd == NULL) {
            SDL_Log("AcquireGPUCommandBuffer failed: %s", SDL_GetError());
            return -1;
        }

        // 1. compute pass
        SDL_Log("Begin procedural texture pass");
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

        //TODO: zero out the counter buffer
        SDL_Log("Begin reset counter pass");
        SDL_GPUComputePass* resetPass = SDL_BeginGPUComputePass(
            cmd,
            nullptr,
            0,
            (SDL_GPUStorageBufferReadWriteBinding[]){
                {
                    .buffer = visibleCounterBuffer,
                }
            },
            1
        );
        SDL_BindGPUComputePipeline(resetPass, resetCounterPipeline);
        // SDL_BindGPUComputeStorageBuffers(resetPass, 0, &counterBuffer, 1);
        SDL_DispatchGPUCompute(resetPass, 1, 1, 1);
        SDL_EndGPUComputePass(resetPass);

        // 2. culling pass
        SDL_Log("Begin culling pass");
        SDL_GPUComputePass* cullingPass = SDL_BeginGPUComputePass(
            cmd,
            nullptr,
            0,
            (SDL_GPUStorageBufferReadWriteBinding[]){
                {
                    .buffer = instanceBuffer,
                },
                {
                    .buffer = visibilityBuffer,
                },
                {
                    .buffer = visibleCounterBuffer,
                }
            },
            3
        );
        SDL_BindGPUComputePipeline(cullingPass, cullingPipeline);
        CameraInfo camInfo = {
            .view = camera.GetViewMatrix(),
            .proj = camera.GetProjMatrix()
        };
        SDL_PushGPUComputeUniformData(cmd, 0, &camInfo, sizeof(CameraInfo));
        SDL_BindGPUComputeStorageBuffers(cullingPass, 0, &instanceBuffer, 1);
        // SDL_BindGPUComputeStorageBuffers(cullingPass, 1, &visibilityBuffer, 1);
        // SDL_BindGPUComputeStorageBuffers(cullingPass, 2, &counterBuffer, 1);
        SDL_DispatchGPUCompute(cullingPass, (instances.size() + 63) / 64, 1, 1);
        SDL_EndGPUComputePass(cullingPass);

        // 3. command building pass
        SDL_Log("Begin command building pass");
        SDL_GPUComputePass* commandBuildingPass = SDL_BeginGPUComputePass(
            cmd,
            nullptr,
            0,
            (SDL_GPUStorageBufferReadWriteBinding[]){
                {
                    .buffer = instanceBuffer,
                },
                {
                    .buffer = visibilityBuffer,
                },
                {
                    .buffer = visibleCounterBuffer,
                },
                {
                    .buffer = drawCommandBuffer,
                }
            },
            4
        );
        SDL_BindGPUComputePipeline(commandBuildingPass, commandBuildingPipeline);
        SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 0, &instanceBuffer, 1);
        SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 1, &visibilityBuffer, 1);
        SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 2, &visibleCounterBuffer, 1);
        // SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 3, &drawCommandBuffer, 1);
        SDL_DispatchGPUCompute(commandBuildingPass, (drawCommands.size() + 63) / 64, 1, 1);
        SDL_EndGPUComputePass(commandBuildingPass);

        // 4. prefix sum pass
        // TODO: reset the buffer every frame?
        SDL_Log("Begin prefix sum pass");
        SDL_GPUComputePass* prefixSumPass = SDL_BeginGPUComputePass(
            cmd,
            nullptr,
            0,
            (SDL_GPUStorageBufferReadWriteBinding[]){
                {
                    .buffer = drawCommandBuffer,
                },
                {
                    .buffer = prefixSumBuffer,
                }
            },
            2
        );
        SDL_BindGPUComputePipeline(prefixSumPass, prefixSumPipeline);
        SDL_DispatchGPUCompute(prefixSumPass, 1, 1, 1);
        SDL_EndGPUComputePass(prefixSumPass);

        // 5. screen pass
        SDL_Log("Begin screen pass");
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
            {
                // Draw cube
                SDL_BindGPUGraphicsPipeline(renderPass, useWireframeMode ? linePipeline : fillPipeline);
                if (useSmallViewport) SDL_SetGPUViewport(renderPass, &SmallViewport);
                if (useScissorRect) SDL_SetGPUScissor(renderPass, &ScissorRect);
                SDL_GPUTextureSamplerBinding textureSamplerBinding = { .texture = procTexture, .sampler = sampler };
                SDL_BindGPUFragmentSamplers(renderPass, 0, &textureSamplerBinding, 1);
                float angle = time * 0.5f;
                CameraInfo camInfo = {
                    .view = camera.GetViewMatrix(),
                    .proj = camera.GetProjMatrix()
                };
                auto instance = testMeshInstances[0];
                instance.model = glm::rotate(glm::identity<glm::mat4>(), angle, glm::vec3(0.0f, 1.0f, -1.0f));
                SDL_PushGPUVertexUniformData(cmd, 0, &camInfo, sizeof(CameraInfo));
                SDL_PushGPUVertexUniformData(cmd, 1, &instance, sizeof(SimpleInstance));

                testMesh.Bind(renderPass);
                if (testMesh.has_indices()) {
                    SDL_DrawGPUIndexedPrimitives(renderPass, testMesh.index_count(), 1, 0, 0, 0);
                } else {
                    SDL_DrawGPUPrimitives(renderPass, testMesh.vertex_count(), 1, 0, 0);
                }

                // Draw SSBO scene
                SDL_BindGPUGraphicsPipeline(renderPass, uberPipeline);
                camInfo = {
                    .view = camera.GetViewMatrix(),
                    .proj = camera.GetProjMatrix()
                };
                SDL_PushGPUVertexUniformData(cmd, 0, &camInfo, sizeof(CameraInfo));
                SDL_BindGPUVertexStorageBuffers(renderPass, 0, (SDL_GPUBuffer*[]){ instanceBuffer, meshBuffer, materialBuffer, vertexBuffer, indexBuffer }, 5);
                SDL_BindGPUFragmentStorageBuffers(renderPass, 0, &materialBuffer, 1);


                // Draw Sponza
                // std::function<void(const std::shared_ptr<Node>&, const glm::mat4&)> drawNode =
                //     [&](const std::shared_ptr<Node>& node, const glm::mat4& parentMatrix) {
                //         if (!node) return;
                //         glm::mat4 localToWorldMatrix = parentMatrix * node->localTransform;
                //         if (node->mesh) {
                //             Transform xform = {
                //                 .model = localToWorldMatrix,
                //                 .view = camera.GetViewMatrix(),
                //                 .proj = camera.GetProjMatrix()
                //             };
                //             SDL_PushGPUVertexUniformData(cmd, 0, &xform, sizeof(Transform));
                //             for (const auto& subMesh : node->mesh->subMeshes) {
                //                 if (subMesh->material) {
                //                     const auto& material = subMesh->material;
                //                     const auto& pipeline = material->GetPipeline(device, renderTargetFormat, msaaSampleCount);
                //                     if (material->pipeline == nullptr) {
                //                         continue;
                //                     }
                //                     SDL_BindGPUGraphicsPipeline(renderPass, pipeline.get());
                //                     for (const auto& vbo : subMesh->vbos) {
                //                         SDL_GPUBufferBinding vertexBufferBinding = { .buffer = vbo.get(), .offset = 0 };
                //                         SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBufferBinding, 1);
                //                     }
                //                     if (subMesh->material->albedoMap) {
                //                         SDL_GPUTextureSamplerBinding textureSamplerBinding = { .texture = subMesh->material->albedoMap.get(), .sampler = sampler };
                //                         SDL_BindGPUFragmentSamplers(renderPass, 0, &textureSamplerBinding, 1);
                //                     }
                //                 } else {
                //                     // TODO: use default material
                //                     continue;
                //                 }
                //                 if (subMesh->ebo) {
                //                     SDL_GPUBufferBinding indexBufferBinding = { .buffer = subMesh->ebo.get(), .offset = 0 };
                //                     SDL_BindGPUIndexBuffer(renderPass, &indexBufferBinding, subMesh->indexType);
                //                     SDL_DrawGPUIndexedPrimitives(renderPass, subMesh->indexCount, 1, 0, 0, 0);
                //                 } else {
                //                     SDL_DrawGPUPrimitives(renderPass, subMesh->vertexCount, 1, 0, 0);
                //                 }
                //             }
                //         }
                //         for (const auto& child : node->children) {
                //             drawNode(child, localToWorldMatrix);
                //         }
                //     };
                // for (const auto& node : sponzaScene->nodes) {
                //     drawNode(node, glm::scale(glm::mat4(1.0f), glm::vec3(100.0f)));
                // }
            }
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
    sponza->Release(device);
    testMesh.Release(device);

    SDL_ReleaseGPUBuffer(device, instanceBuffer);
    SDL_ReleaseGPUBuffer(device, meshBuffer);
    SDL_ReleaseGPUBuffer(device, materialBuffer);
    SDL_ReleaseGPUBuffer(device, visibilityBuffer);
    SDL_ReleaseGPUBuffer(device, visibleCounterBuffer);
    SDL_ReleaseGPUBuffer(device, drawCommandBuffer);
    SDL_ReleaseGPUBuffer(device, prefixSumBuffer);
    SDL_ReleaseGPUBuffer(device, vertexBuffer);
    SDL_ReleaseGPUBuffer(device, indexBuffer);

    SDL_ReleaseGPUComputePipeline(device, procTexturePipeline);
    SDL_ReleaseGPUComputePipeline(device, resetCounterPipeline);
    SDL_ReleaseGPUComputePipeline(device, cullingPipeline);
    SDL_ReleaseGPUComputePipeline(device, commandBuildingPipeline);
    SDL_ReleaseGPUComputePipeline(device, prefixSumPipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, fillPipeline);
	SDL_ReleaseGPUGraphicsPipeline(device, linePipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, uberPipeline);

    SDL_ReleaseGPUTexture(device, imgTexture);
    SDL_ReleaseGPUTexture(device, procTexture);
    SDL_ReleaseGPUSampler(device, sampler);
    SDL_ReleaseGPUTexture(device, msaaTexture);
    SDL_ReleaseGPUTexture(device, resolveTexture);

    // Release window and GPU device
    SDL_ReleaseWindowFromGPUDevice(device, window);
    SDL_DestroyWindow(window);
	SDL_DestroyGPUDevice(device);
    SDL_Quit();

    return 0;
}