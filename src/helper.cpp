#include "helper.hpp"
#include "SDL_gpu_shadercross.h"
#include "scene.hpp"
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>

#define STBI_NO_SIMD
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"


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

Scene* LoadGLTF(SDL_GPUDevice* device, const char* filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool result = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty()) {
        SDL_Log("GLTF Warning: %s", warn.c_str());
    }
    if (!err.empty()) {
        SDL_Log("GLTF Error: %s", err.c_str());
    }
    if (!result) {
        SDL_Log("Failed to parse glTF");
        return nullptr;
    }

    if (model.scenes.empty()) {
        SDL_Log("No scenes found in gltf");
        return nullptr;
    }

    Scene* scene = new Scene();

    const auto GetLocalMatrix = [](const tinygltf::Node &node) -> glm::mat4{
        if (!node.matrix.empty()) {
            return glm::mat4(
                node.matrix[0], node.matrix[1], node.matrix[2], node.matrix[3],
                node.matrix[4], node.matrix[5], node.matrix[6], node.matrix[7],
                node.matrix[8], node.matrix[9], node.matrix[10], node.matrix[11],
                node.matrix[12], node.matrix[13], node.matrix[14], node.matrix[15]
            );
        }
        const auto translation =
            node.translation.empty()
            ? glm::mat4(1.0f)
            : glm::translate(glm::mat4(1.0f), glm::vec3(node.translation[0], node.translation[1], node.translation[2]));;
        const auto rotationQuat =
            node.rotation.empty()
            ? glm::quat(1, 0, 0, 0)
            : glm::quat(float(node.rotation[3]), float(node.rotation[0]), float(node.rotation[1]),float(node.rotation[2]));
        const auto TR = translation * glm::mat4_cast(rotationQuat);
        return node.scale.empty()
            ? TR
            : glm::scale(TR, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
    };

    const auto GetBufferSize = [](const tinygltf::Accessor& accessor) -> size_t {
        size_t bufferSize = accessor.count;
        switch (accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                bufferSize *= 1;
                break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                bufferSize *= 2;
                break;
            case TINYGLTF_COMPONENT_TYPE_INT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            case TINYGLTF_COMPONENT_TYPE_FLOAT:
                bufferSize *= 4;
                break;
            default:
                SDL_Log("Unknown component type");
                break;
        }
        switch (accessor.type) {
            case TINYGLTF_TYPE_SCALAR: return bufferSize;
            case TINYGLTF_TYPE_VEC2: return bufferSize * 2;
            case TINYGLTF_TYPE_VEC3: return bufferSize * 3;
            case TINYGLTF_TYPE_VEC4: return bufferSize * 4;
            case TINYGLTF_TYPE_MAT2: return bufferSize * 4;
            case TINYGLTF_TYPE_MAT3: return bufferSize * 9;
            case TINYGLTF_TYPE_MAT4: return bufferSize * 16;
            default:
                SDL_Log("Unknown type");
                return 0;
        }
    };

    const auto CreateAndUploadBuffer = [device](const void* data, size_t size) -> std::shared_ptr<SDL_GPUBuffer> {
        SDL_GPUBufferCreateInfo bufferCreateInfo = {
            .usage = SDL_GPU_BUFFERUSAGE_VERTEX | SDL_GPU_BUFFERUSAGE_INDEX,
            .size = static_cast<Uint32>(size)
        };
        auto buffer = std::shared_ptr<SDL_GPUBuffer>(
            SDL_CreateGPUBuffer(device, &bufferCreateInfo),
            [device](SDL_GPUBuffer* buf) { SDL_ReleaseGPUBuffer(device, buf); }
        );
        SDL_GPUTransferBufferCreateInfo transferInfo = {
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .size = static_cast<Uint32>(size)
        };
        SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(device, &transferInfo);

        void* transferData = SDL_MapGPUTransferBuffer(device, transferBuffer, false);
        memcpy(transferData, data, size);
        SDL_UnmapGPUTransferBuffer(device, transferBuffer);

        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);

        SDL_GPUTransferBufferLocation src = {
            .transfer_buffer = transferBuffer,
            .offset = 0
        };
        SDL_GPUBufferRegion dst = {
            .buffer = buffer.get(),
            .offset = 0,
            .size = static_cast<Uint32>(size)
        };
        SDL_UploadToGPUBuffer(copyPass, &src, &dst, false);

        SDL_EndGPUCopyPass(copyPass);
        SDL_SubmitGPUCommandBuffer(cmd);

        SDL_ReleaseGPUTransferBuffer(device, transferBuffer);

        return buffer;
    };

    const auto CreateAndUploadTexture = [device](const tinygltf::Image& img) -> std::shared_ptr<SDL_GPUTexture> {
        SDL_GPUTextureCreateInfo texCreateInfo = {
            .type = SDL_GPU_TEXTURETYPE_2D,
            .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
            .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
            .width = static_cast<Uint32>(img.width),
            .height = static_cast<Uint32>(img.height),
            .layer_count_or_depth = 1,
            .num_levels = 1,
        };
        auto texture = std::shared_ptr<SDL_GPUTexture>(
            SDL_CreateGPUTexture(device, &texCreateInfo),
            [device](SDL_GPUTexture* tex) { SDL_ReleaseGPUTexture(device, tex); }
        );
        SDL_GPUTransferBufferCreateInfo transferInfo = {
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .size = static_cast<Uint32>(img.image.size())
        };
        SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(device, &transferInfo);

        void* transferData = SDL_MapGPUTransferBuffer(device, transferBuffer, false);
        memcpy(transferData, img.image.data(), img.image.size());
        SDL_UnmapGPUTransferBuffer(device, transferBuffer);

        SDL_GPUCommandBuffer* cmd = SDL_AcquireGPUCommandBuffer(device);
        SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(cmd);

        SDL_GPUTextureTransferInfo src = {
            .transfer_buffer = transferBuffer,
            .offset = 0
        };
        SDL_GPUTextureRegion dst = {
            .texture = texture.get(),
            .layer = 0,
            .w = static_cast<Uint32>(img.width),
            .h = static_cast<Uint32>(img.height),
            .d = 1
        };
        SDL_UploadToGPUTexture(copyPass, &src, &dst, false);

        SDL_EndGPUCopyPass(copyPass);
        SDL_SubmitGPUCommandBuffer(cmd);

        SDL_ReleaseGPUTransferBuffer(device, transferBuffer);

        return texture;
    };

    std::vector<std::shared_ptr<Material>> materials;
    materials.reserve(model.materials.size());
    for (const auto& mat : model.materials) {
        auto material = std::make_shared<Material>();
        std::vector<int> texIndices = {
            mat.pbrMetallicRoughness.baseColorTexture.index,
            mat.normalTexture.index,
            mat.pbrMetallicRoughness.metallicRoughnessTexture.index,
            mat.occlusionTexture.index,
            mat.emissiveTexture.index,
        };
        for (int texIdx : texIndices) {
            if (texIdx >= 0) {
                const auto& srcTex = model.textures[texIdx];
                auto texture = CreateAndUploadTexture(model.images[srcTex.source]);
                if (texIdx == mat.pbrMetallicRoughness.baseColorTexture.index) {
                    material->albedoMap = texture;
                } else if (texIdx == mat.normalTexture.index) {
                    material->normalMap = texture;
                } else if (texIdx == mat.pbrMetallicRoughness.metallicRoughnessTexture.index) {
                    material->metallicRoughnessMap = texture;
                } else if (texIdx == mat.occlusionTexture.index) {
                    material->occlusionMap = texture;
                } else if (texIdx == mat.emissiveTexture.index) {
                    material->emissiveMap = texture;
                }
            }
        }
        materials.push_back(material);
    }

    std::vector<std::shared_ptr<Mesh>> meshes;
    meshes.reserve(model.meshes.size());
    for (const auto& srcMesh : model.meshes) {
        auto mesh = std::make_shared<Mesh>();
        mesh->name = srcMesh.name;

        for (const auto& primitive : srcMesh.primitives) {
            auto subMesh = std::make_unique<SubMesh>();
            // TODO: set subMesh->mode
            const std::array<std::string, 3> attributeNames = { "POSITION", "NORMAL", "TEXCOORD_0" };
            for (const auto& attr : attributeNames) {
                auto it = primitive.attributes.find(attr);
                if (it != primitive.attributes.end()) {
                    const auto& accessor = model.accessors[it->second];
                    const auto& bufferView = model.bufferViews[accessor.bufferView];
                    const auto& buffer = model.buffers[bufferView.buffer];
                    const auto bufferSize = GetBufferSize(accessor);
                    auto vbo = CreateAndUploadBuffer(
                        buffer.data.data() + bufferView.byteOffset + accessor.byteOffset,
                        bufferSize
                    );
                    subMesh->vbos.push_back(vbo);
                    // All attributes in a primitive must have the same number of vertices according to the glTF spec
                    subMesh->vertexCount = accessor.count;
                }
            }
            if (primitive.material >= 0) {
                subMesh->material = materials[primitive.material];
                // TODO: pipeline creation
                // SDL_GPUGraphicsPipelineCreateInfo pipelineInfo = {};
            }
            if (primitive.indices >= 0) {
                const auto& accessor = model.accessors[primitive.indices];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                subMesh->ebo = CreateAndUploadBuffer(
                    buffer.data.data() + bufferView.byteOffset + accessor.byteOffset,
                    GetBufferSize(accessor)
                );
                subMesh->indexCount = accessor.count;
                // TODO: set subMesh->indexType
            }
            mesh->subMeshes.push_back(std::move(subMesh));
        }

        meshes.push_back(mesh);
    }

    std::function<std::shared_ptr<Node>(int)> createNode = [&](int nodeIndex) -> std::shared_ptr<Node> {
        const auto& srcNode = model.nodes[nodeIndex];
        auto node = std::make_shared<Node>();
        node->name = srcNode.name;
        node->localTransform = GetLocalMatrix(srcNode);

        if (srcNode.mesh >= 0) {
            node->mesh = meshes[srcNode.mesh];
        }
        for (int childIdx : srcNode.children) {
            node->children.push_back(createNode(childIdx));
        }

        return node;
    };

    const auto& srcScene = model.defaultScene >= 0 ? model.scenes[model.defaultScene] : model.scenes[0];
    for (int nodeIdx : srcScene.nodes) {
        scene->nodes.push_back(createNode(nodeIdx));
    }
    scene->name = srcScene.name;
    return scene;
}