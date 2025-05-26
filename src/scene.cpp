#include "scene.hpp"
#include "SDL3/SDL_log.h"

void Scene::Draw() {

}

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

void Scene::Unload() {
    for (const auto& node : nodes) {
        if (node->mesh) {
            for (const auto& subMesh : node->mesh->subMeshes) {
                //subMesh->material->
            }
        }
    }
}