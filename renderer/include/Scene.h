#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include <Buffer.h>
#include <Primitives/SkyBox.h>
#include <Lights/Lights.h>

namespace quasar {

class Scene {
public:
    struct GPUPointLightBlock {
        PointLight::GPUPointLight lights[PointLight::maxPointLights];
        int numPointLights;
    };

    glm::vec4 backgroundColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    SkyBox* envCubeMap = nullptr;

    AmbientLight* ambientLight = nullptr;
    DirectionalLight* directionalLight = nullptr;
    std::vector<PointLight*> pointLights;

    Node rootNode;

    Scene() = default;
    ~Scene() = default;

    void setEnvMap(SkyBox* envCubeMap);
    void setAmbientLight(AmbientLight* ambientLight);
    void setDirectionalLight(DirectionalLight* directionalLight);

    void addPointLight(PointLight* pointLight);
    void addChildNode(Node* node);

    void updateAnimations(float dt);

    int bindMaterial(const Material* material, Buffer& pointLightsUBO);

    Node* findNodeByName(const std::string& name);

    void clear();

    static const uint numTextures = 3;

private:
    GPUPointLightBlock pointLightsData;

    int bindAmbientLight(const Material* material, int texIdx);
    int bindDirectionalLight(const Material* material, int texIdx);
    int bindPointLights(const Material* material, Buffer& pointLightsUBO, int texIdx);
};

} // namespace quasar

#endif // SCENE_H
