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

std::map<SDL_GPUGraphicsPipeline*, std::shared_ptr<Material>> pipelineCache;

SDL_GPUComputePipeline* CreateComputePipelineFromShader(
    SDL_GPUDevice* device,
    const char* filename,
    Uint32 samplerCount,
    Uint32 uniformBufferCount,
    Uint32 readonlyStorageBufferCount,
    Uint32 readonlyStorageTextureCount,
	Uint32 readwriteStorageBufferCount,
    Uint32 readwriteStorageTextureCount,
    Uint32 threadCountX,
    Uint32 threadCountY,
    Uint32 threadCountZ
) {
    std::string filePath = std::string(SDL_GetBasePath()) + "res/shaders/" + filename + ".spv";

	size_t codeSize;
	void* code = SDL_LoadFile(filePath.c_str(), &codeSize);
	if (code == nullptr) {
		SDL_Log("Failed to load compute shader from disk! %s", filePath.c_str());
		return nullptr;
	}

    SDL_GPUComputePipelineCreateInfo compPipelineDesc = {
        .code_size = codeSize,
        .code = static_cast<const Uint8 *>(code),
        .entrypoint = "main",
        .format = SDL_GPU_SHADERFORMAT_SPIRV,
        .num_samplers = samplerCount,
        .num_uniform_buffers = uniformBufferCount,
        .num_readonly_storage_buffers = readonlyStorageBufferCount,
        .num_readonly_storage_textures = readonlyStorageTextureCount,
        .num_readwrite_storage_buffers = readwriteStorageBufferCount,
        .num_readwrite_storage_textures = readwriteStorageTextureCount,
        .threadcount_x = threadCountX,
        .threadcount_y = threadCountY,
        .threadcount_z = threadCountZ,
    };

	SDL_GPUComputePipeline* pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(device, &compPipelineDesc);
	if (pipeline == nullptr) {
		SDL_Log("Failed to create compute pipeline!");
		SDL_free(code);
		return nullptr;
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
		return nullptr;
	}

    std::string filePath = std::string(SDL_GetBasePath()) + "res/shaders/" + filename + ".spv";

	size_t codeSize;
	void* code = SDL_LoadFile(filePath.c_str(), &codeSize);
	if (code == nullptr) {
		SDL_Log("Failed to load shader from disk! %s", filePath.c_str());
		return nullptr;
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
	if (shader == nullptr) {
		SDL_Log("Failed to create shader!");
		SDL_free(code);
		return nullptr;
	}

	SDL_free(code);
	return shader;
}

std::shared_ptr<Image> LoadImage(const char* filename) {
    std::string filePath = std::string(SDL_GetBasePath()) + "res/" + filename;

    int width, height, numChannels;
    if (!stbi_info(filePath.c_str(), &width, &height, &numChannels)) {
        SDL_Log("Failed to load image at %s!\n", filePath.c_str());
        return nullptr;
    }
    SDL_PixelFormat desiredFormat;
    switch (numChannels) {
    case 3:
    case 4:
        desiredFormat = SDL_PixelFormat::SDL_PIXELFORMAT_RGBA32;
        break;
    default:
        SDL_Log("Unknown texture format at %s\n", filePath.c_str());
        return nullptr;
    }

    uint8_t* data = stbi_load(filePath.c_str(), &width, &height, &numChannels, 4);
    if (!data) {
        SDL_Log("Failed to load image data at %s!\n", filePath.c_str());
        return nullptr;
    }
    auto image = std::make_shared<Image>(Image {
        .uri = filename,
        .width = static_cast<Uint32>(width),
        .height = static_cast<Uint32>(height),
        .component = 4,
        .pixels = std::vector<Uint8>(data, data + (width * height * 4))
    });
    stbi_image_free(data);
    return image;
}

std::shared_ptr<Scene> LoadGLTF(SDL_GPUDevice* device, const char* filename) {
    std::string filePath = std::string(SDL_GetBasePath()) + "res/" + filename;

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool result = loader.LoadASCIIFromFile(&model, &err, &warn, filePath.c_str());
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

    std::shared_ptr<Scene> scene = std::make_shared<Scene>();

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

    const auto GetBufferElementSize = [](const tinygltf::Accessor& accessor) -> size_t {
        size_t elmSize = 0;
        switch (accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                elmSize = 1;
                break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                elmSize = 2;
                break;
            case TINYGLTF_COMPONENT_TYPE_INT:
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
            case TINYGLTF_COMPONENT_TYPE_FLOAT:
                elmSize = 4;
                break;
            default:
                SDL_Log("Unknown component type");
                break;
        }
        switch (accessor.type) {
            case TINYGLTF_TYPE_SCALAR: return elmSize;
            case TINYGLTF_TYPE_VEC2: return elmSize * 2;
            case TINYGLTF_TYPE_VEC3: return elmSize * 3;
            case TINYGLTF_TYPE_VEC4: return elmSize * 4;
            case TINYGLTF_TYPE_MAT2: return elmSize * 4;
            case TINYGLTF_TYPE_MAT3: return elmSize * 9;
            case TINYGLTF_TYPE_MAT4: return elmSize * 16;
            default:
                SDL_Log("Unknown type");
                return 0;
        }
    };

    const auto GetBufferElementFormat = [](const tinygltf::Accessor& accessor) -> SDL_GPUVertexElementFormat {
        switch (accessor.componentType) {
            case TINYGLTF_COMPONENT_TYPE_INT:
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INT;
                } else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INT2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INT3;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INT4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_UINT;
                } else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_UINT2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_UINT3;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_UINT4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            case TINYGLTF_COMPONENT_TYPE_FLOAT:
                if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_FLOAT;
                } else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_UBYTE2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            case TINYGLTF_COMPONENT_TYPE_BYTE:
                if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_BYTE2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_BYTE4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            case TINYGLTF_COMPONENT_TYPE_SHORT:
                if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_SHORT2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_SHORT4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                if (accessor.type == TINYGLTF_TYPE_VEC2) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_USHORT2;
                } else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                    return SDL_GPU_VERTEXELEMENTFORMAT_USHORT4;
                } else {
                    return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
                }
            default:
                SDL_Log("Unknown component type");
                return SDL_GPU_VERTEXELEMENTFORMAT_INVALID;
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

    // Load images
    std::vector<std::shared_ptr<Image>> images;
    images.reserve(model.images.size());
    for (const auto& img : model.images) {
        images.push_back(std::make_shared<Image>(Image {
            .uri = img.uri,
            .width = static_cast<Uint32>(img.width),
            .height = static_cast<Uint32>(img.height),
            .component = static_cast<Uint32>(img.component),
            .pixels = std::vector<Uint8>(img.image.begin(), img.image.end())
        }));
    }

    // Load materials
    std::vector<std::shared_ptr<Material>> materials;
    materials.reserve(model.materials.size());
    for (const auto& mat : model.materials) {
        auto material = std::make_shared<Material>();
        material->name = mat.name;
        if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            const auto& texture = model.textures[mat.pbrMetallicRoughness.baseColorTexture.index];
            if (texture.source >= 0) {
                material->albedoMap = images[texture.source];
                // material->uvs["albedo"] = mat.pbrMetallicRoughness.baseColorTexture.texCoord;
                // material->samplers["albedo"] = texture.sampler;
            }
        }
        if (mat.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
            const auto& texture = model.textures[mat.pbrMetallicRoughness.metallicRoughnessTexture.index];
            if (texture.source >= 0) {
                material->metallicRoughnessMap = images[texture.source];
                // material->uvs["metallicRoughness"] = mat.pbrMetallicRoughness.metallicRoughnessTexture.texCoord;
                // material->samplers["metallicRoughness"] = texture.sampler;
            }
        }
        if (mat.normalTexture.index >= 0) {
            const auto& texture = model.textures[mat.normalTexture.index];
            if (texture.source >= 0) {
                material->normalMap = images[texture.source];
            }
        }
        if (mat.occlusionTexture.index >= 0) {
            const auto& texture = model.textures[mat.occlusionTexture.index];
            if (texture.source >= 0) {
                material->occlusionMap = images[texture.source];
            }
        }
        if (mat.emissiveTexture.index >= 0) {
            const auto& texture = model.textures[mat.emissiveTexture.index];
            if (texture.source >= 0) {
                material->emissiveMap = images[texture.source];
            }
        }
        // pipeline creation
        SDL_GPUGraphicsPipelineCreateInfo pipelineInfo = {
            .vertex_shader = LoadShader(device, "TBN.vert", 0, 2, 0, 0),
            .fragment_shader = LoadShader(device, "PBR.frag", 5, 1, 0, 0),
            .rasterizer_state = {
                .fill_mode = SDL_GPU_FILLMODE_FILL,
                .cull_mode = SDL_GPU_CULLMODE_BACK,
            }
        };
        material->pipelineInfo = pipelineInfo;
        materials.push_back(material);
    }

    // Load meshes
    std::vector<std::shared_ptr<MeshGroup>> meshGroups;
    meshGroups.reserve(model.meshes.size());
    for (const auto& srcMesh : model.meshes) {
        auto meshGroup = std::make_shared<MeshGroup>();
        meshGroup->name = srcMesh.name;

        for (const auto& primitive : srcMesh.primitives) {
            bool invalid = false;
            auto mesh = std::make_unique<Mesh>();
            for (const auto& attr : primitive.attributes) {
                const auto& accessor = model.accessors[attr.second];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];
                const auto elmSize = GetBufferElementSize(accessor);
                const auto elmFormat = GetBufferElementFormat(accessor);
                const float* data = reinterpret_cast<const float*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);

                if (attr.first == "POSITION") {
                    mesh->positions.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->positions[i] = glm::vec3(
                            data[i * 3 + 0],
                            -data[i * 3 + 1], // Y is inverted
                            data[i * 3 + 2]
                        );
                    }
                }
                else if (attr.first == "NORMAL") {
                    mesh->normals.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->normals[i] = glm::vec3(
                            data[i * 3 + 0],
                            -data[i * 3 + 1], // Y is inverted
                            data[i * 3 + 2]
                        );
                    }
                } else if (attr.first == "TEXCOORD_0") {
                    mesh->uv0s.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->uv0s[i] = glm::vec2(
                            data[i * 2 + 0],
                            data[i * 2 + 1]
                        );
                    }
                } else if (attr.first == "TEXCOORD_1") {
                    mesh->uv1s.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->uv1s[i] = glm::vec2(
                            data[i * 2 + 0],
                            data[i * 2 + 1]
                        );
                    }
                } else if (attr.first == "TANGENT") {
                    mesh->tangents.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->tangents[i] = glm::vec3(
                            data[i * 3 + 0],
                            -data[i * 3 + 1], // Y is inverted
                            data[i * 3 + 2]
                        );
                    }
                } else if (attr.first == "COLOR_0") {
                    mesh->colors.resize(accessor.count);
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->colors[i] = glm::vec4(
                            data[i * 4 + 0],
                            data[i * 4 + 1],
                            data[i * 4 + 2],
                            data[i * 4 + 3]
                        );
                    }
                }
            }
            // std::vector<SDL_GPUVertexBufferDescription> vertexBufferDescs;
            // std::vector<SDL_GPUVertexAttribute> vertexAttributes;
            // Uint32 attrLocation = 0;
            // const std::array<std::string, 3> attributeNames = { "POSITION", "NORMAL", "TEXCOORD_0" };
            // for (const auto& attr : attributeNames) {
            //     auto it = primitive.attributes.find(attr);
            //     if (it != primitive.attributes.end()) {
            //         const auto& accessor = model.accessors[it->second];
            //         const auto& bufferView = model.bufferViews[accessor.bufferView];
            //         const auto& buffer = model.buffers[bufferView.buffer];
            //         const auto elmSize = GetBufferElementSize(accessor);
            //         const auto elmFormat = GetBufferElementFormat(accessor);
            //         if (elmFormat <= SDL_GPU_VERTEXELEMENTFORMAT_INVALID) {
            //             SDL_Log("Invalid element format for attribute %s", attr.c_str());
            //             invalid = true;
            //             break;
            //         }
            //         const auto bufferSize = accessor.count * elmSize;
            //         // TODO: check if the buffer is already created, if so, just reuse it and uodate specified bytes
            //         // Currently, a new buffer is created for each attribute
            //         auto vbo = CreateAndUploadBuffer(
            //             buffer.data.data() + bufferView.byteOffset + accessor.byteOffset,
            //             bufferSize
            //         );
            //         vertexBufferDescs.push_back(SDL_GPUVertexBufferDescription {
            //             .slot = static_cast<Uint32>(mesh->vbos.size()),
            //             .pitch = static_cast<Uint32>(bufferSize),
            //             .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
            //             .instance_step_rate = 0,
            //         });
            //         vertexAttributes.push_back(SDL_GPUVertexAttribute {
            //             .location = attrLocation++,
            //             .buffer_slot = static_cast<Uint32>(mesh->vbos.size()),
            //             .format = elmFormat,
            //             .offset = 0
            //         });
            //         mesh->vbos.push_back(vbo);
            //         // All attributes in a primitive must have the same number of vertices according to the glTF spec
            //         mesh->vertexCount = accessor.count;
            //     }
            // }
            // if (invalid) {
            //     SDL_Log("Skipping a primitive due to invalid format");
            //     continue;
            // }
            if (primitive.indices >= 0) {
                const auto& accessor = model.accessors[primitive.indices];
                const auto& bufferView = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[bufferView.buffer];

                // mesh->ebo = CreateAndUploadBuffer(
                //     buffer.data.data() + bufferView.byteOffset + accessor.byteOffset,
                //     accessor.count * GetBufferElementSize(accessor)
                // );
                // mesh->indexCount = accessor.count;
                // mesh->indexType = SDL_GPU_INDEXELEMENTSIZE_16BIT;

                mesh->indices.resize(accessor.count);

                switch (accessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    const Uint16* data = reinterpret_cast<const Uint16*>(
                        &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                    );
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->indices[i] = static_cast<Uint32>(data[i]);
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    const Uint32* data = reinterpret_cast<const Uint32*>(
                        &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                    );
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->indices[i] = data[i];
                    }
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    const Uint8* data = reinterpret_cast<const Uint8*>(
                        &buffer.data[bufferView.byteOffset + accessor.byteOffset]
                    );
                    for (size_t i = 0; i < accessor.count; i++) {
                        mesh->indices[i] = static_cast<Uint32>(data[i]);
                    }
                    break;
                }
                default:
                    SDL_Log("Unsupported index component type: %d", accessor.componentType);
                    break;
                }
            }
            if (primitive.material >= 0) {
                mesh->material = materials[primitive.material];
            } else {
                // if no material is specified, mesh->material would be nullptr
                SDL_Log("No material specified for primitive");
            }
            switch (primitive.mode) {
            case TINYGLTF_MODE_POINTS:
                mesh->primitiveMode = PrimitiveMode::POINTS;
                break;
            case TINYGLTF_MODE_LINE:
                mesh->primitiveMode = PrimitiveMode::LINES;
                break;
            case TINYGLTF_MODE_LINE_STRIP:
                mesh->primitiveMode = PrimitiveMode::LINE_STRIP;
                break;
            case TINYGLTF_MODE_TRIANGLES:
                mesh->primitiveMode = PrimitiveMode::TRIANGLES;
                break;
            case TINYGLTF_MODE_TRIANGLE_STRIP:
                mesh->primitiveMode = PrimitiveMode::TRIANGLE_STRIP;
                break;
            default:
                throw std::runtime_error("Unsupported primitive mode");
            }
            meshGroup->meshes.push_back(std::move(mesh));
        }

        meshGroups.push_back(meshGroup);
    }

    std::function<std::shared_ptr<Node>(int)> createNode = [&](int nodeIndex) -> std::shared_ptr<Node> {
        const auto& srcNode = model.nodes[nodeIndex];
        auto node = std::make_shared<Node>();
        node->name = srcNode.name;
        node->localTransform = GetLocalMatrix(srcNode);

        if (srcNode.mesh >= 0) {
            node->meshGroup = meshGroups[srcNode.mesh];
        }
        for (int childIdx : srcNode.children) {
            node->children.push_back(createNode(childIdx));
        }

        return node;
    };

    const auto& srcScene = model.defaultScene >= 0 ? model.scenes[model.defaultScene] : model.scenes[0];
    scene->name = srcScene.name;
    // TODO: maybe directly store the images and materials in the scene
    scene->images = std::move(images);
    scene->materials = std::move(materials);
    for (int nodeIdx : srcScene.nodes) {
        scene->nodes.push_back(createNode(nodeIdx));
    }
    return scene;
}