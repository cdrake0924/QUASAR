#include <Cameras/VRCamera.h>

using namespace quasar;

VRCamera::VRCamera()
    : VRCamera(DEFAULT_FOV_DEG, DEFAULT_ASPECT, DEFAULT_NEAR, DEFAULT_FAR)
{}

VRCamera::VRCamera(float fovyDeg, float aspect, float near, float far)
    : left(PerspectiveCamera(fovyDeg, aspect, near, far))
    , right(PerspectiveCamera(fovyDeg, aspect, near, far))
{
    left .setPosition({ -DEFAULT_IPD / 2, 0.0f, 0.0f });
    right.setPosition({ +DEFAULT_IPD / 2, 0.0f, 0.0f });

    addChildNode(&left);
    addChildNode(&right);
}

VRCamera::VRCamera(const glm::uvec2& windowSize)
    : VRCamera(DEFAULT_FOV_DEG, (float)windowSize.x / (float)windowSize.y, DEFAULT_NEAR, DEFAULT_FAR)
{}

VRCamera::VRCamera(uint width, uint height)
    : VRCamera(DEFAULT_FOV_DEG, (float)width / (float)height, DEFAULT_NEAR, DEFAULT_FAR)
{}

VRCamera::VRCamera(const glm::mat4 (&projs)[2])
    : left(projs[0])
    , right(projs[1])
{
    addChildNode(&left);
    addChildNode(&right);
}

void VRCamera::setProjectionMatrix(const glm::mat4& proj) {
    left.setProjectionMatrix(proj);
    right.setProjectionMatrix(proj);
}

glm::mat4 VRCamera::getProjectionMatrix() const {
    return left.getProjectionMatrix();
}

void VRCamera::updateProjectionMatrix() {
    left.updateProjectionMatrix();
    right.updateProjectionMatrix();
}

glm::mat4 VRCamera::getEyeViewMatrix(bool isLeftEye) const {
    if (isLeftEye) {
        return left.getViewMatrix();
    }
    else {
        return right.getViewMatrix();
    }
}

void VRCamera::setProjectionMatrix(float fovy, float aspect, float near, float far) {
    left.setProjectionMatrix(fovy, aspect, near, far);
    right.setProjectionMatrix(fovy, aspect, near, far);
}

void VRCamera::setProjectionMatrices(const glm::mat4 (&projs)[2]) {
    left.setProjectionMatrix(projs[0]);
    right.setProjectionMatrix(projs[1]);
}

void VRCamera::setViewMatrices(const glm::mat4 (&views)[2]) {
    left.setViewMatrix(views[0]);
    right.setViewMatrix(views[1]);
}
