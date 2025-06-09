#include "scene.hpp"
#include "SDL3/SDL_gpu.h"
#include "SDL3/SDL_log.h"


void Scene::Print() {
    SDL_Log("Scene %s", name.c_str());
    SDL_Log("--------------------------------");
    for (const auto& node : nodes) {
        PrintNode(node);
    }
}

void Scene::PrintNode(const std::shared_ptr<Node>& node) {
    SDL_Log("Node %s", node->name.c_str());
    SDL_Log("--------------------------------");
    if (node->meshGroup) {
        SDL_Log("meshes: %d", static_cast<int>(node->meshGroup->meshes.size()));
        for (const auto& mesh : node->meshGroup->meshes) {
            SDL_Log("Vertex count: %zu", mesh->positions.size());
            SDL_Log("Normal count: %zu", mesh->normals.size());
            SDL_Log("UV count: %zu", mesh->uv0s.size());
            if (mesh->indices.size() > 0) {
                SDL_Log("Index count: %zu", mesh->indices.size());
                for (const Uint32& idx : mesh->indices) {
                    SDL_Log(
                        "(Vertex %u) Position: %f, %f, %f, UV: %f, %f, Normal: %f, %f, %f",
                        idx,
                        mesh->positions[idx].x,
                        mesh->positions[idx].y,
                        mesh->positions[idx].z,
                        mesh->uv0s[idx].x,
                        mesh->uv0s[idx].y,
                        mesh->normals[idx].x,
                        mesh->normals[idx].y,
                        mesh->normals[idx].z
                    );
                }
            }
        }
    }
    SDL_Log("--------------------------------");
    for (const auto& child : node->children) {
        PrintNode(child);
    }
}

void Scene::Upload(SDL_GPUDevice* device) {
    for (const auto& image : images) {
        image->texture = CreateTexture(image, device);
    }
    for (const auto& material : materials) {
        material->pipelineInfo.target_info = {
            .color_target_descriptions = (SDL_GPUColorTargetDescription[]){{
                .format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
            }},
            .num_color_targets = 1,
        };
        // TODO: create pipeline variants for different vertex layouts
        material->pipelineInfo.vertex_input_state = (SDL_GPUVertexInputState){
            .vertex_buffer_descriptions = (SDL_GPUVertexBufferDescription[]){
                {
                    .slot = 0,
                    .pitch = sizeof(glm::vec3),
                    .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                    .instance_step_rate = 0,
                },
                {
                    .slot = 1,
                    .pitch = sizeof(glm::vec2),
                    .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                    .instance_step_rate = 0,
                },
                {
                    .slot = 2,
                    .pitch = sizeof(glm::vec3),
                    .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                    .instance_step_rate = 0,
                },
                {
                    .slot = 3,
                    .pitch = sizeof(glm::vec3),
                    .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
                    .instance_step_rate = 0,
                },
            },
            .num_vertex_buffers = 4,
            .vertex_attributes = (SDL_GPUVertexAttribute[]){
                {
                    .location = 0,
                    .buffer_slot = 0,
                    .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
                    .offset = 0,
                },
                {
                    .location = 1,
                    .buffer_slot = 1,
                    .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
                    .offset = 0,
                },
                {
                    .location = 2,
                    .buffer_slot = 2,
                    .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
                    .offset = 0,
                },
                {
                    .location = 3,
                    .buffer_slot = 3,
                    .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
                    .offset = 0,
                },
            },
            .num_vertex_attributes = 4,
        };
        material->pipelineInfo.multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_4
        };

        material->pipeline = std::shared_ptr<SDL_GPUGraphicsPipeline>(
            SDL_CreateGPUGraphicsPipeline(device, &material->pipelineInfo),
            [device](SDL_GPUGraphicsPipeline* pipeline) { SDL_ReleaseGPUGraphicsPipeline(device, pipeline); }
        );
        if (material->pipeline == nullptr) {
            SDL_Log("Failed to create pipeline: %s", SDL_GetError());
        }
    }
    for (const auto& node : nodes) {
        UploadNode(node, device);
    }
}

void Scene::UploadNode(const std::shared_ptr<Node>& node, SDL_GPUDevice* device) {
    if (node->meshGroup) {
        for (const auto& mesh : node->meshGroup->meshes) {
            if (mesh->primitiveMode != PrimitiveMode::TRIANGLES) {
                // TODO: create pipeline variants for different primitive modes
                SDL_Log("Primitive mode is not triangle list");
                continue;
            }

            const auto& positions = mesh->positions;
            const auto& uvs = mesh->uv0s;
            const auto& normals = mesh->normals;
            const auto& tangents = mesh->tangents;
            mesh->vbos.push_back(CreateBuffer(positions.data(), positions.size() * sizeof(glm::vec3), SDL_GPU_BUFFERUSAGE_VERTEX, device));
            mesh->vbos.push_back(CreateBuffer(uvs.data(), uvs.size() * sizeof(glm::vec2), SDL_GPU_BUFFERUSAGE_VERTEX, device));
            mesh->vbos.push_back(CreateBuffer(normals.data(), normals.size() * sizeof(glm::vec3), SDL_GPU_BUFFERUSAGE_VERTEX, device));
            if (tangents.size() > 0) {
                mesh->vbos.push_back(CreateBuffer(tangents.data(), tangents.size() * sizeof(glm::vec3), SDL_GPU_BUFFERUSAGE_VERTEX, device));
            }

            const auto& indices = mesh->indices;
            if (indices.size() > 0) {
                mesh->ebo = CreateBuffer(indices.data(), indices.size() * sizeof(Uint32), SDL_GPU_BUFFERUSAGE_INDEX, device);
            }
        }
    }
    for (const auto& child : node->children) {
        UploadNode(child, device);
    }
}

auto Scene::CreateTexture(const std::shared_ptr<Image>& image, SDL_GPUDevice* device) -> std::unique_ptr<SDL_GPUTexture, std::function<void(SDL_GPUTexture*)>> {
    SDL_GPUTextureCreateInfo texCreateInfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = static_cast<Uint32>(image->width),
        .height = static_cast<Uint32>(image->height),
        .layer_count_or_depth = 1,
        .num_levels = 1,
    };
    if (image->component == 1) {
        texCreateInfo.format = SDL_GPU_TEXTUREFORMAT_R8_UNORM;
    } else if (image->component == 2) {
        texCreateInfo.format = SDL_GPU_TEXTUREFORMAT_R8G8_UNORM;
    } else if (image->component == 4) {
        texCreateInfo.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    } else {
        SDL_Log("Unknown component count: %d", image->component);
        return nullptr;
    }
    auto texture = std::unique_ptr<SDL_GPUTexture, std::function<void(SDL_GPUTexture*)>>(
        SDL_CreateGPUTexture(device, &texCreateInfo),
        [device](SDL_GPUTexture* tex) { SDL_ReleaseGPUTexture(device, tex); }
    );
    SDL_GPUTransferBufferCreateInfo transferInfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = static_cast<Uint32>(image->pixels.size())
    };
    SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(device, &transferInfo);

    void* transferData = SDL_MapGPUTransferBuffer(device, transferBuffer, false);
    memcpy(transferData, image->pixels.data(), image->pixels.size());
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
        .w = static_cast<Uint32>(image->width),
        .h = static_cast<Uint32>(image->height),
        .d = 1
    };
    SDL_UploadToGPUTexture(copyPass, &src, &dst, false);

    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(cmd);

    SDL_ReleaseGPUTransferBuffer(device, transferBuffer);

    return texture;
}

auto Scene::CreateBuffer(const void* data, size_t size, SDL_GPUBufferUsageFlags usage, SDL_GPUDevice* device) -> std::shared_ptr<SDL_GPUBuffer> {
    SDL_GPUBufferCreateInfo bufferCreateInfo = {
        .usage = usage,
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
}

std::shared_ptr<Node> Scene::CreateNode(const std::string& name, const glm::mat4& transform) {
    auto node = std::make_shared<Node>();
    node->name = name;
    node->localTransform = transform;
    nodes.push_back(node);
    return node;
}

void Scene::AddNode(std::shared_ptr<Node> node) {
    nodes.push_back(node);
}

std::shared_ptr<Node> Scene::FindNode(const std::string& name) {
    for (const auto& node : nodes) {
        auto result = FindNodeInHierarchy(name, node);
        if (result) {
            return result;
        }
    }
    return nullptr;
}

std::shared_ptr<Node> Scene::FindNodeInHierarchy(const std::string& name, const std::shared_ptr<Node>& node) {
    if (node->name == name) {
        return node;
    }
    for (const auto& childNode : node->children) {
        auto result = FindNodeInHierarchy(name, childNode);
        if (result) {
            return result;
        }
    }
    return nullptr;
}

void Scene::Draw(SDL_GPURenderPass* renderPass) {
    for (const auto& node : nodes) {
        if (node->meshGroup) {
            for (const auto& mesh : node->meshGroup->meshes) {
                SDL_BindGPUGraphicsPipeline(renderPass, mesh->material->pipeline.get());
                std::array<SDL_GPUBufferBinding, 4> vertexBufferBindings = {{
                    { .buffer = mesh->vbos[0].get(), .offset = 0 },
                    { .buffer = mesh->vbos[1].get(), .offset = 0 },
                    { .buffer = mesh->vbos[2].get(), .offset = 0 },
                    { .buffer = mesh->vbos[3].get(), .offset = 0 },
                }};
                SDL_BindGPUVertexBuffers(renderPass, 0, vertexBufferBindings.data(), vertexBufferBindings.size());
                if (mesh->indices.size() > 0) {
                    SDL_GPUBufferBinding indexBufferBinding = { .buffer = mesh->ebo.get(), .offset = 0 };
                    SDL_BindGPUIndexBuffer(renderPass, &indexBufferBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
                }
                if (mesh->indices.size() > 0) {
                    SDL_DrawGPUIndexedPrimitives(renderPass, mesh->indices.size(), 1, 0, 0, 0);
                } else {
                    SDL_DrawGPUPrimitives(renderPass, mesh->positions.size(), 1, 0, 0);
                }
            }
        }
    }
}

void Scene::Release(SDL_GPUDevice* device) {
    // The textures will be released by deleters
    // for (const auto& image : images) {
    //     SDL_ReleaseGPUTexture(device, image->texture.get());
    // }
    // The pipelines will be released by deleters
    for (const auto& material : materials) {
        SDL_ReleaseGPUShader(device, material->pipelineInfo.vertex_shader);
        SDL_ReleaseGPUShader(device, material->pipelineInfo.fragment_shader);
        // SDL_ReleaseGPUGraphicsPipeline(device, material->pipeline.get());
    }
    for (const auto& node : nodes) {
        ReleaseNode(node, device);
    }
    images.clear(); // Triggers texture deleters
    materials.clear(); // Triggers pipeline deleters
    nodes.clear(); // Triggers buffer deleters
}

void Scene::ReleaseNode(const std::shared_ptr<Node>& node, SDL_GPUDevice* device) {
    if (node->meshGroup) {
        for (const auto& mesh : node->meshGroup->meshes) {
            // The buffers will be released by deleters
            // for (const auto& vbo : mesh->vbos) {
            //     SDL_ReleaseGPUBuffer(device, vbo.get());
            // }
            // if (mesh->ebo) {
            //     SDL_ReleaseGPUBuffer(device, mesh->ebo.get());
            // }
        }
    }
    for (const auto& child : node->children) {
        ReleaseNode(child, device);
    }
}

void Scene::Update(float dt) {
    const glm::mat4 rootTransform = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));
    for (const auto& node : nodes) {
        UpdateNode(node, rootTransform);
    }
}

void Scene::UpdateNode(const std::shared_ptr<Node>& node, const glm::mat4& parentTransform) {
    if (node->isTransformDirty) {
        node->worldTransform = parentTransform * node->localTransform;
        node->isTransformDirty = false;
    }
    for (const auto& child : node->children) {
        UpdateNode(child, node->worldTransform);
    }
}

// Usage example
// auto scene = Scene();
// auto entity = scene.CreateNode("Cube", glm::identity<glm::mat4>());