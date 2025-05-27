#include "scene.hpp"
#include "SDL3/SDL_log.h"


void Scene::Print() {
    SDL_Log("scene: %s", name.c_str());
    SDL_Log("nodes: %d", static_cast<int>(nodes.size()));
    for (const auto& node : nodes) {
        SDL_Log("node: %s", node->name.c_str());
        if (node->mesh) {
            SDL_Log("meshes: %d", static_cast<int>(node->mesh->subMeshes.size()));
        }
    }
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

void Scene::Release(SDL_GPUDevice* device) {
    for (const auto& node : nodes) {
        if (node->mesh) {
            for (const auto& subMesh : node->mesh->subMeshes) {
                for (const auto& vbo : subMesh->vbos) {
                    SDL_ReleaseGPUBuffer(device, vbo.get());
                }
                if (subMesh->ebo) {
                    SDL_ReleaseGPUBuffer(device, subMesh->ebo.get());
                }
                // if (subMesh->material) {
                //     if (subMesh->material->albedoMap) {
                //         SDL_ReleaseGPUTexture(device, subMesh->material->albedoMap.get());
                //     }
                //     if (subMesh->material->pipeline) {
                //         SDL_ReleaseGPUGraphicsPipeline(device, subMesh->material->pipeline.get());
                //     }
                // }
            }
        }
    }
}

// Usage example
// auto scene = Scene();
// auto entity = scene.CreateNode("Cube", glm::identity<glm::mat4>());