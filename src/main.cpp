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
#include "rng.hpp"

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
    Sint32 baseVertex;
    Uint32 baseIndex;
    Uint32 indexCount;
};

struct MaterialInfo {
    // Uint8 albedoMap;
    // Uint8 normalMap;
    // Uint8 metallicRoughnessMap;
    // Uint8 aoMap;
    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 specular;
    float roughness;
};

// We're using SDL_GPUIndexedIndirectDrawCommand directly
// struct DrawCommand {
//     Uint32 indexCount;
//     Uint32 instanceCount;
//     Uint32 firstIndex;
//     Uint32 baseVertex;
//     Uint32 baseInstance;
// };

struct Particle {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 velocity;
    alignas(16) glm::vec3 force;
    glm::vec4 color;
};

// TODO: unsure about alignment
struct ShaderParams {
    glm::vec2 resolution;
    glm::vec2 mousePosition;
    float time;
    float deltaTime;
};

int windowWidth, windowHeight;

int main(int argc, char* args[]) {
    RNG rng;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL could not initialize! Error: %s\n", SDL_GetError());
        return -1;
    }

    SDL_Window* window = SDL_CreateWindow("SDL3 GPU Demo", 720, 720, SDL_WINDOW_RESIZABLE);
    SDL_GetWindowSizeInPixels(window, &windowWidth, &windowHeight);

    SDL_GPUDevice* device = SDL_CreateGPUDevice(SDL_ShaderCross_GetSPIRVShaderFormats(), true, NULL);
	if (device == NULL) {
		SDL_Log("GPUCreateDevice failed: %s", SDL_GetError());
		return -1;
	}
	if (!SDL_ClaimWindowForGPUDevice(device, window)) {
		fmt::print("GPUClaimWindow failed");
		return -1;
	}

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

    SDL_GPUShader* particleVertexShader = LoadShader(device, "Particle.vert", 0, 1, 1, 0);
	if (particleVertexShader == NULL) {
		SDL_Log("Failed to create vertex shader!");
		return -1;
	}

	SDL_GPUShader* particleFragmentShader = LoadShader(device, "Particle.frag", 0, 0, 0, 0);
	if (particleFragmentShader == NULL) {
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

    SDL_GPUComputePipeline* particleSingleBufferPipeline = CreateComputePipelineFromShader(
	    device,
	    "ParticleSingleBuffer.comp",
        0, 1, 0, 0, 1, 0, 256, 1, 1
    );

    SDL_GPUComputePipeline* particleForcePipeline = CreateComputePipelineFromShader(
	    device,
	    "ParticleForce.comp",
        0, 2, 0, 0, 1, 0, 256, 1, 1
    );

    SDL_GPUComputePipeline* particleIntegratePipeline = CreateComputePipelineFromShader(
	    device,
	    "ParticleIntegrate.comp",
        0, 2, 0, 0, 1, 0, 256, 1, 1
    );

    SDL_GPUComputePipeline* cullingPipeline = CreateComputePipelineFromShader(
	    device,
	    "Cull.comp",
        0, 1, 1, 0, 2, 0, 64, 1, 1
    );

    SDL_GPUComputePipeline* commandBuildingPipeline = CreateComputePipelineFromShader(
	    device,
	    "CommandBuild.comp",
        0, 0, 4, 0, 1, 0, 64, 1, 1
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
				.format = renderTargetFormat,
                .blend_state = {
				    .enable_blend = true,
					.alpha_blend_op = SDL_GPU_BLENDOP_ADD,
					.color_blend_op = SDL_GPU_BLENDOP_ADD,
					.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
					.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
					.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE,
					.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE
				}
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
    gfxPipelineDesc.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
    gfxPipelineDesc.vertex_shader = particleVertexShader;
    gfxPipelineDesc.fragment_shader = particleFragmentShader;
    SDL_GPUGraphicsPipeline* particlePipeline = SDL_CreateGPUGraphicsPipeline(device, &gfxPipelineDesc);
    if (particlePipeline == NULL) {
        SDL_Log("Failed to create particle pipeline!");
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
	SDL_ReleaseGPUShader(device, particleVertexShader);
	SDL_ReleaseGPUShader(device, particleFragmentShader);

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
    // std::shared_ptr<Scene> sponza = LoadGLTF(device, "res/models/Sponza/Sponza.gltf");
    // sponza->Print();
    auto simpleCube = CPUMesh::CreateCube();
    auto simpleCubeInstances = std::vector<SimpleInstance>({
        {
            .model = glm::identity<glm::mat4>()
        }
    });
    auto quad = CPUMesh::CreateQuad();

    constexpr Uint32 numParticles = 2000000;
    std::vector<Particle> particles(numParticles);
    for (auto& particle : particles) {
        float r = sqrt(rng.RandomFloat()) * 0.1f;
        float theta = rng.RandomFloat() * 2 * glm::pi<float>();
        float spiral_offset = rng.RandomFloat() * 2.5f;
        float x = r * cos(theta + spiral_offset) * windowHeight / windowWidth;
        float y = r * sin(theta + spiral_offset);

        particle.position = glm::vec3(x, y, 0);
        glm::vec3 tangent = glm::normalize(glm::vec3(-y, x, 0));
        particle.velocity = tangent * (0.2f / (r + 0.025f)) * 0.25f;

        float brightness = 1.0f - (r / 1.0f);
        // glm::vec3 a = glm::vec3(0.746, 0.815, 0.846);
        // glm::vec3 b = glm::vec3(0.195, 0.283, 0.187);
        // glm::vec3 c = glm::vec3(1.093, 1.417, 1.405);
        // glm::vec3 d = glm::vec3(5.435, 2.400, 5.741);
        glm::vec3 a = glm::vec3(0.427, 0.346, 0.372);
        glm::vec3 b = glm::vec3(0.288, 0.918, 0.336);
        glm::vec3 c = glm::vec3(0.635, 1.136, 0.404);
        glm::vec3 d = glm::vec3(1.893, 0.663, 1.910);
        particle.color = glm::vec4(a + b * glm::cos(6.28318f * (c * brightness + d)), 1.0f);
    }

    // Data for SSBOs
    // FIXME: setting instance count to 1000000 is causing seg fault
    constexpr Uint32 numInstances = 200;
    constexpr Uint32 numMeshes = 10;
    constexpr Uint32 numMaterials = 100;
    constexpr Uint32 numDrawCommands = numMeshes;
    auto cubeVertices = CreateCubeVertices();
    auto cubeIndices = CreateCubeIndices();
    auto sphereVertices = CreateSphereVertices();
    auto sphereIndices = CreateSphereIndices();
    std::array<Instance, numInstances> instances = {};
    std::array<MeshInfo, numMeshes> meshInfos = {};
    std::array<MaterialInfo, numMaterials> materialInfos = {};
    std::array<Uint32, numInstances> visibleInstanceIndices = {};
    Uint32 visibleCounter = 0;
    std::array<SDL_GPUIndexedIndirectDrawCommand, numDrawCommands> drawCommands = {};
    std::array<Uint32, numDrawCommands> prefixSums = {};

    for (int i = 0; i < instances.size(); i++) {
        instances[i] = {
            .model = glm::translate(glm::identity<glm::mat4>(), glm::vec3(rng.RandomFloatInRange(-10.0f, 10.0f), rng.RandomFloatInRange(-10.0f, 10.0f), rng.RandomFloatInRange(-10.0f, 10.0f))),
            .meshID = static_cast<Uint32>(rng.RandomIntInRange(0, meshInfos.size() - 1)),
            .materialID = static_cast<Uint32>(rng.RandomIntInRange(0, materialInfos.size() - 1))
        };
    }
    for (int i = 0; i < meshInfos.size(); i++) {
        if (rng.RandomFloat() < 0.5f) {
            meshInfos[i] = {
                .baseVertex = 0,
                .baseIndex = 0,
                .indexCount = 36
            };
        } else {
            meshInfos[i] = {
                .baseVertex = 24,
                .baseIndex = 36,
                .indexCount = 1584
            };
        }
    }
    for (int i = 0; i < materialInfos.size(); i++) {
        materialInfos[i] = {
            .diffuse = glm::vec3(rng.RandomFloat(), rng.RandomFloat(), rng.RandomFloat()),
            .specular = glm::vec3(rng.RandomFloat(), rng.RandomFloat(), rng.RandomFloat()),
            .roughness = rng.RandomFloat()
        };
    }

    Uint32 vertexBufferSize = sizeof(Vertex) * cubeVertices.size() + sizeof(Vertex) * sphereVertices.size();
    Uint32 indexBufferSize = sizeof(Uint32) * cubeIndices.size() + sizeof(Uint32) * sphereIndices.size();
    Uint32 instanceBufferSize = sizeof(Instance) * instances.size();
    Uint32 meshBufferSize = sizeof(MeshInfo) * meshInfos.size();
    Uint32 materialBufferSize = sizeof(MaterialInfo) * materialInfos.size();
    Uint32 visibilityBufferSize = sizeof(Uint32) * visibleInstanceIndices.size();
    Uint32 visibleCounterBufferSize = sizeof(Uint32);
    Uint32 drawCommandBufferSize = sizeof(SDL_GPUIndexedIndirectDrawCommand) * drawCommands.size();
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

    // Create buffers
    SDL_GPUBufferCreateInfo particleBufferDesc = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE | SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ,
        .size = static_cast<Uint32>(sizeof(Particle) * particles.size())
    };
    SDL_GPUBuffer* particleBuffer0 = SDL_CreateGPUBuffer(device, &particleBufferDesc);
    SDL_GPUBuffer* particleBuffer1 = SDL_CreateGPUBuffer(device, &particleBufferDesc);

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
    SDL_GPUTransferBufferCreateInfo cubeTransferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = static_cast<Uint32>(simpleCube.vertex_byte_count() * simpleCube.vertex_count() + simpleCube.index_byte_count() * simpleCube.index_count())
    };
    SDL_GPUTransferBuffer* cubeTransferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &cubeTransferBufferCreateInfo
	);
    simpleCube.Stage(device, cubeTransferBuffer);

    SDL_GPUTransferBufferCreateInfo quadTransferBufferCreateInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = static_cast<Uint32>(quad.vertex_byte_count() * quad.vertex_count() + quad.index_byte_count() * quad.index_count())
    };
    SDL_GPUTransferBuffer* quadTransferBuffer = SDL_CreateGPUTransferBuffer(
		device,
        &quadTransferBufferCreateInfo
	);
    quad.Stage(device, quadTransferBuffer);

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

    SDL_GPUTransferBufferCreateInfo particleTransferBufferDesc = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = static_cast<Uint32>(sizeof(Particle) * particles.size())
    };
    SDL_GPUTransferBuffer* particleTransferBuffer = SDL_CreateGPUTransferBuffer(
        device,
        &particleTransferBufferDesc
    );
    auto particleTransferData = reinterpret_cast<Particle*>(
        SDL_MapGPUTransferBuffer(
            device,
            particleTransferBuffer,
            false
        )
    );
    memcpy(particleTransferData, particles.data(), sizeof(Particle) * particles.size()); // Leave it mapped
    SDL_UnmapGPUTransferBuffer(device, particleTransferBuffer);

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

    simpleCube.Upload(device, copyPass, cubeTransferBuffer);

    quad.Upload(device, copyPass, quadTransferBuffer);

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

    SDL_GPUTransferBufferLocation particleTransferLocation = {
        .transfer_buffer = particleTransferBuffer,
        .offset = 0
    };
    SDL_GPUBufferRegion particleTransferRegion = {
        .buffer = particleBuffer0,
        .offset = 0,
        .size = static_cast<Uint32>(sizeof(Particle) * particles.size())
    };
    SDL_UploadToGPUBuffer(
        copyPass,
        &particleTransferLocation,
        &particleTransferRegion,
        false
    );
    particleTransferRegion.buffer = particleBuffer1;
    SDL_UploadToGPUBuffer(
        copyPass,
        &particleTransferLocation,
        &particleTransferRegion,
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

    SDL_ReleaseGPUTransferBuffer(device, cubeTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, quadTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, texTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, ssboTransferBuffer);
    SDL_ReleaseGPUTransferBuffer(device, particleTransferBuffer);

    Camera camera(
        glm::vec3(0.0f, 0.0f, -5.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::radians(120.0f),
        windowWidth / (float)windowHeight,
        0.1f,
        500.0f
    );

    // Main loop
    Uint32 frameCount = 0;
    float time = SDL_GetTicks() / 1000.0f;
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

        float currTime = SDL_GetTicks() / 1000.0f;
        float deltaTime = currTime - time;
        time = currTime;

        float mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);

        ShaderParams shaderParams = {
            .resolution = glm::vec2(windowWidth, windowHeight),
            .time = time,
            .deltaTime = deltaTime,
            .mousePosition = glm::vec2(mouseX, mouseY)
        };
        CameraInfo camInfo = {
            .view = camera.GetViewMatrix(),
            .proj = camera.GetProjMatrix()
        };

        glm::vec2 uv = shaderParams.mousePosition / shaderParams.resolution;
        glm::vec3 ndc = glm::vec3(uv.x, uv.y, 1.0) * 2.0f - 1.0f;
        ndc.y = -ndc.y;
        glm::vec4 viewPosH = glm::inverse(camInfo.proj) * glm::vec4(ndc, 1.0);
        glm::vec3 viewPos = glm::vec3(viewPosH) / viewPosH.w;
        glm::vec3 worldPos = glm::vec3(glm::inverse(camInfo.view) * glm::vec4(viewPos, 1.0));
        glm::vec3 ro = camera.GetEye();
        glm::vec3 rd = glm::normalize(worldPos - ro);
        glm::vec3 attractorPos = ro + rd * 5.0f;

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
        SDL_PushGPUComputeUniformData(cmd, 0, &shaderParams, sizeof(ShaderParams));
        SDL_DispatchGPUCompute(computePass, ceil(procTextureSize / 16.0), ceil(procTextureSize / 16.0), 1);
        SDL_EndGPUComputePass(computePass);

        SDL_GenerateMipmapsForGPUTexture(cmd, procTexture);

        // SDL_GPUComputePass* particleSingleBufferPass = SDL_BeginGPUComputePass(
        //     cmd,
        //     nullptr,
        //     0,
        //     (SDL_GPUStorageBufferReadWriteBinding[]){
        //         {
        //             .buffer = particleBuffer0,
        //         },
        //     },
        //     1
        // );
        // SDL_BindGPUComputePipeline(particleSingleBufferPass, particleSingleBufferPipeline);
        // SDL_PushGPUComputeUniformData(cmd, 0, &shaderParams, sizeof(ShaderParams));
        // SDL_DispatchGPUCompute(particleSingleBufferPass, (particles.size() + 255) / 256, 1, 1);
        // SDL_EndGPUComputePass(particleSingleBufferPass);

        SDL_GPUComputePass* particleForcePass = SDL_BeginGPUComputePass(
            cmd,
            nullptr,
            0,
            (SDL_GPUStorageBufferReadWriteBinding[]){
                {
                    .buffer = particleBuffer0,
                }
            },
            1
        );
        SDL_BindGPUComputePipeline(particleForcePass, particleForcePipeline);
        SDL_PushGPUComputeUniformData(cmd, 0, &shaderParams, sizeof(ShaderParams));
        SDL_PushGPUComputeUniformData(cmd, 1, &attractorPos, sizeof(glm::vec3));
        SDL_DispatchGPUCompute(particleForcePass, (particles.size() + 255) / 256, 1, 1);
        SDL_EndGPUComputePass(particleForcePass);

        SDL_GPUComputePass* particleIntegratePass = SDL_BeginGPUComputePass(
            cmd,
            nullptr,
            0,
            (SDL_GPUStorageBufferReadWriteBinding[]){
                {
                    .buffer = particleBuffer0,
                }
            },
            1
        );
        SDL_BindGPUComputePipeline(particleIntegratePass, particleIntegratePipeline);
        SDL_PushGPUComputeUniformData(cmd, 0, &shaderParams, sizeof(ShaderParams));
        SDL_PushGPUComputeUniformData(cmd, 1, &attractorPos, sizeof(glm::vec3));
        SDL_DispatchGPUCompute(particleIntegratePass, (particles.size() + 255) / 256, 1, 1);
        SDL_EndGPUComputePass(particleIntegratePass);

        // //TODO: zero out the counter buffer
        // SDL_Log("Begin reset counter pass");
        // SDL_GPUComputePass* resetPass = SDL_BeginGPUComputePass(
        //     cmd,
        //     nullptr,
        //     0,
        //     (SDL_GPUStorageBufferReadWriteBinding[]){
        //         {
        //             .buffer = visibleCounterBuffer,
        //         }
        //     },
        //     1
        // );
        // SDL_BindGPUComputePipeline(resetPass, resetCounterPipeline);
        // // SDL_BindGPUComputeStorageBuffers(resetPass, 0, &counterBuffer, 1);
        // SDL_DispatchGPUCompute(resetPass, 1, 1, 1);
        // SDL_EndGPUComputePass(resetPass);

        // // 2. culling pass
        // SDL_Log("Begin culling pass");
        // SDL_GPUComputePass* cullingPass = SDL_BeginGPUComputePass(
        //     cmd,
        //     nullptr,
        //     0,
        //     (SDL_GPUStorageBufferReadWriteBinding[]){
        //         {
        //             .buffer = instanceBuffer,
        //         },
        //         {
        //             .buffer = visibilityBuffer,
        //         },
        //         {
        //             .buffer = visibleCounterBuffer,
        //         }
        //     },
        //     3
        // );
        // SDL_BindGPUComputePipeline(cullingPass, cullingPipeline);
        // CameraInfo camInfo = {
        //     .view = camera.GetViewMatrix(),
        //     .proj = camera.GetProjMatrix()
        // };
        // SDL_PushGPUComputeUniformData(cmd, 0, &camInfo, sizeof(CameraInfo));
        // SDL_BindGPUComputeStorageBuffers(cullingPass, 0, &instanceBuffer, 1);
        // // SDL_BindGPUComputeStorageBuffers(cullingPass, 1, &visibilityBuffer, 1);
        // // SDL_BindGPUComputeStorageBuffers(cullingPass, 2, &visibleCounterBuffer, 1);
        // SDL_DispatchGPUCompute(cullingPass, (instances.size() + 63) / 64, 1, 1);
        // SDL_EndGPUComputePass(cullingPass);

        // // 3. command building pass
        // SDL_Log("Begin command building pass");
        // SDL_GPUComputePass* commandBuildingPass = SDL_BeginGPUComputePass(
        //     cmd,
        //     nullptr,
        //     0,
        //     (SDL_GPUStorageBufferReadWriteBinding[]){
        //         {
        //             .buffer = instanceBuffer,
        //         },
        //         {
        //             .buffer = visibilityBuffer,
        //         },
        //         {
        //             .buffer = visibleCounterBuffer,
        //         },
        //         {
        //             .buffer = meshBuffer,
        //         },
        //         {
        //             .buffer = drawCommandBuffer,
        //         }
        //     },
        //     5
        // );
        // SDL_BindGPUComputePipeline(commandBuildingPass, commandBuildingPipeline);
        // SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 0, &instanceBuffer, 1);
        // SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 1, &visibilityBuffer, 1);
        // SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 2, &visibleCounterBuffer, 1);
        // SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 3, &meshBuffer, 1);
        // // SDL_BindGPUComputeStorageBuffers(commandBuildingPass, 4, &drawCommandBuffer, 1);
        // SDL_DispatchGPUCompute(commandBuildingPass, (drawCommands.size() + 63) / 64, 1, 1);
        // SDL_EndGPUComputePass(commandBuildingPass);

        // // 4. prefix sum pass
        // // TODO: reset the buffer every frame?
        // SDL_Log("Begin prefix sum pass");
        // SDL_GPUComputePass* prefixSumPass = SDL_BeginGPUComputePass(
        //     cmd,
        //     nullptr,
        //     0,
        //     (SDL_GPUStorageBufferReadWriteBinding[]){
        //         {
        //             .buffer = drawCommandBuffer,
        //         },
        //         {
        //             .buffer = prefixSumBuffer,
        //         }
        //     },
        //     2
        // );
        // SDL_BindGPUComputePipeline(prefixSumPass, prefixSumPipeline);
        // SDL_DispatchGPUCompute(prefixSumPass, 1, 1, 1);
        // SDL_EndGPUComputePass(prefixSumPass);

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
                auto instance = simpleCubeInstances[0];
                instance.model = glm::rotate(glm::identity<glm::mat4>(), angle, glm::vec3(0.0f, 1.0f, -1.0f));
                SDL_PushGPUVertexUniformData(cmd, 0, &camInfo, sizeof(CameraInfo));
                SDL_PushGPUVertexUniformData(cmd, 1, &instance, sizeof(SimpleInstance));

                simpleCube.Bind(renderPass);
                // if (simpleCube.has_indices()) {
                //     SDL_DrawGPUIndexedPrimitives(renderPass, simpleCube.index_count(), 1, 0, 0, 0);
                // } else {
                //     SDL_DrawGPUPrimitives(renderPass, simpleCube.vertex_count(), 1, 0, 0);
                // }

                // Draw particles
                SDL_BindGPUGraphicsPipeline(renderPass, particlePipeline);
                SDL_PushGPUVertexUniformData(cmd, 0, &camInfo, sizeof(CameraInfo));
                SDL_BindGPUVertexStorageBuffers(renderPass, 0, &particleBuffer0, 1);
                quad.Bind(renderPass);
                if (quad.has_indices()) {
                    SDL_DrawGPUIndexedPrimitives(renderPass, quad.index_count(), particles.size(), 0, 0, 0);
                } else {
                    SDL_DrawGPUPrimitives(renderPass, quad.vertex_count(), particles.size(), 0, 0);
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
                // SDL_DrawGPUIndexedPrimitivesIndirect(renderPass, drawCommandBuffer, 0, drawCommands.size());

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
                // for (const auto& node : sponza->nodes) {
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

        frameCount++;
    }

    // Release GPU resources
    // sponza->Release(device);
    simpleCube.Release(device);
    quad.Release(device);

    SDL_ReleaseGPUBuffer(device, instanceBuffer);
    SDL_ReleaseGPUBuffer(device, meshBuffer);
    SDL_ReleaseGPUBuffer(device, materialBuffer);
    SDL_ReleaseGPUBuffer(device, visibilityBuffer);
    SDL_ReleaseGPUBuffer(device, visibleCounterBuffer);
    SDL_ReleaseGPUBuffer(device, drawCommandBuffer);
    SDL_ReleaseGPUBuffer(device, prefixSumBuffer);
    SDL_ReleaseGPUBuffer(device, vertexBuffer);
    SDL_ReleaseGPUBuffer(device, indexBuffer);
    SDL_ReleaseGPUBuffer(device, particleBuffer0);
    SDL_ReleaseGPUBuffer(device, particleBuffer1);

    SDL_ReleaseGPUComputePipeline(device, procTexturePipeline);
    SDL_ReleaseGPUComputePipeline(device, particleForcePipeline);
    SDL_ReleaseGPUComputePipeline(device, particleIntegratePipeline);
    SDL_ReleaseGPUComputePipeline(device, particleSingleBufferPipeline);
    SDL_ReleaseGPUComputePipeline(device, resetCounterPipeline);
    SDL_ReleaseGPUComputePipeline(device, cullingPipeline);
    SDL_ReleaseGPUComputePipeline(device, commandBuildingPipeline);
    SDL_ReleaseGPUComputePipeline(device, prefixSumPipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, fillPipeline);
	SDL_ReleaseGPUGraphicsPipeline(device, linePipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, uberPipeline);
    SDL_ReleaseGPUGraphicsPipeline(device, particlePipeline);

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