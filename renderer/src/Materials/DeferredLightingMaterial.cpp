#include <Lights/PointLight.h>
#include <Materials/DeferredLightingMaterial.h>

using namespace quasar;

std::shared_ptr<Shader> DeferredLightingMaterial::shader = nullptr;

DeferredLightingMaterial::DeferredLightingMaterial() {
    if (shader == nullptr) {
        ShaderDataCreateParams dirShadowMapParams{
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DEFERRED_LIGHTING_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DEFERRED_LIGHTING_FRAG_len,
            .defines = {
                "#define MAX_POINT_LIGHTS " + std::to_string(PointLight::maxPointLights),
            }
        };
        shader = std::make_shared<Shader>(dirShadowMapParams);
    }
}

void DeferredLightingMaterial::bindGBuffer(const GBuffer& gBuffer) const {
    shader->setTexture("gAlbedo", gBuffer.albedoTexture, 0);
    shader->setTexture("gPBR", gBuffer.pbrTexture, 1);
    shader->setTexture("gAlpha", gBuffer.alphaTexture, 2);
    shader->setTexture("gNormal", gBuffer.normalsTexture, 3);
    shader->setTexture("gPosition", gBuffer.positionTexture, 4);
    shader->setTexture("gLightPosition", gBuffer.lightPositionTexture, 5);
}

void DeferredLightingMaterial::bindCamera(const Camera& camera) const {
    shader->setVec3("camera.position", camera.getPosition());
}
