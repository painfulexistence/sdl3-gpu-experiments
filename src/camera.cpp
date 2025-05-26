#include "camera.hpp"
#include <glm/trigonometric.hpp>

void Camera::Dolly(float offset) {
    _eye += glm::vec3(0.0f, 0.0f, offset);
    _center += glm::vec3(0.0f, 0.0f, offset);
    _isViewDirty = true;
}

void Camera::Truck(float offset) {
    _eye += glm::vec3(offset, 0.0f, 0.0f);
    _center += glm::vec3(offset, 0.0f, 0.0f);
    _isViewDirty = true;
}

void Camera::Pedestal(float offset) {
    _eye += glm::vec3(0.0f, offset, 0.0f);
    _center += glm::vec3(0.0f, offset, 0.0f);
    _isViewDirty = true;
}

void Camera::Pan(float radians) {
    // float radius = glm::length(_eye - _center);
    // _center = _eye + glm::vec3(radius * std::cos(radians), radius * std::sin(radians), 0.0f);
    // _isViewDirty = true;
}

void Camera::Tilt(float radians) {
    // float radius = glm::length(_eye - _center);
    // _center = _eye + glm::vec3(radius * std::cos(radians), radius * std::sin(radians), 0.0f);
    // _isViewDirty = true;
}

void Camera::Roll(float radians) {
}

void Camera::Orbit(float radians) {

}

void Camera::UpdateAspectRatio(float aspect) {
    _aspect = aspect;
    _isProjDirty = true;
}