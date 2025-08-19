#include <Cameras/PerspectiveCamera.h>

using namespace quasar;

PerspectiveCamera::PerspectiveCamera()
    : PerspectiveCamera(DEFAULT_FOV_DEG, DEFAULT_ASPECT, DEFAULT_NEAR, DEFAULT_FAR)
{}

PerspectiveCamera::PerspectiveCamera(float fovyDeg, float aspect, float near, float far) {
    setProjectionMatrix(glm::radians(fovyDeg), aspect, near, far);
    updateCameraOrientation();
}

PerspectiveCamera::PerspectiveCamera(uint width, uint height)
    : PerspectiveCamera(DEFAULT_FOV_DEG, (float)width / (float)height, DEFAULT_NEAR, DEFAULT_FAR)
{}

PerspectiveCamera::PerspectiveCamera(const glm::uvec2& windowSize)
    : PerspectiveCamera(DEFAULT_FOV_DEG, (float)windowSize.x / (float)windowSize.y, DEFAULT_NEAR, DEFAULT_FAR)
{}


PerspectiveCamera::PerspectiveCamera(const glm::mat4& proj) {
    setProjectionMatrix(proj);
    updateCameraOrientation();
}

void PerspectiveCamera::setProjectionMatrix(const glm::mat4& proj) {
    this->proj = proj;
    this->projInverse = glm::inverse(proj);

    fovyRad = atan(1.0f / proj[1][1]) * 2.0f;
    aspect = proj[1][1] / proj[0][0];
    near = proj[3][2] / (proj[2][2] - 1.0f);
    far = proj[3][2] / (proj[2][2] + 1.0f);
}

void PerspectiveCamera::setProjectionMatrix(float fovyRad, float aspect, float near, float far) {
    this->fovyRad = fovyRad;
    this->aspect = aspect;
    this->near = near;
    this->far = far;
    updateProjectionMatrix();
}

void PerspectiveCamera::updateProjectionMatrix() {
    proj = glm::perspective(fovyRad, aspect, near, far);
    projInverse = glm::inverse(proj);

    frustum.setFromCameraMatrices(view, proj);
}

void PerspectiveCamera::setViewMatrix(const glm::mat4& view) {
    this->view = view;
    viewInverse = glm::inverse(view);

    setTransformLocalFromParent(view);
    updateCameraOrientation();
    frustum.setFromCameraMatrices(view, proj);
}

void PerspectiveCamera::updateViewMatrix() {
    view = getTransformLocalFromParent();
    viewInverse = glm::inverse(view);

    updateCameraOrientation();
    frustum.setFromCameraMatrices(view, proj);
}

void PerspectiveCamera::processScroll(float yoffset) {
    movementSpeed += yoffset * scrollSensitivity;
    movementSpeed = glm::clamp(movementSpeed, 0.1f, 20.0f);
}

void PerspectiveCamera::processKeyboard(const Keys& keys, double deltaTime) {
    float velocity = movementSpeed * deltaTime;
    if (keys.W_PRESSED)
        position += front * velocity;
    if (keys.A_PRESSED)
        position -= right * velocity;
    if (keys.S_PRESSED)
        position -= front * velocity;
    if (keys.D_PRESSED)
        position += right * velocity;
    if (keys.Q_PRESSED)
        position -= up * velocity;
    if (keys.E_PRESSED)
        position += up * velocity;

    updateViewMatrix();
}

void PerspectiveCamera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
    yaw   += xoffset * mouseSensitivity;
    pitch += yoffset * mouseSensitivity;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch) {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }

    setOrientationFromYawPitch();
}

void PerspectiveCamera::updateCameraOrientation() {
    front = -glm::normalize(glm::vec3(view[0][2], view[1][2], view[2][2]));
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));

    yaw = glm::degrees(atan2(front.z, front.x));
    pitch = glm::degrees(asin(front.y));
}

void PerspectiveCamera::setOrientationFromYawPitch() {
    // Calculate the new front vector
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);

    // Also re-calculate the right and up vector
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));

    glm::mat4 newView = glm::lookAt(position, position + front, up);
    setTransformParentFromLocal(glm::inverse(newView));

    updateViewMatrix();
}
