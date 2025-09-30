#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <Scene.h>
#include <Cameras/PerspectiveCamera.h>
#include <Primitives/Primitives.h>

namespace quasar {

class SceneLoader {
public:
    std::vector<Mesh*> meshes;
    std::map<std::string, int> meshIndices;

    std::vector<Model*> models;
    std::map<std::string, int> modelIndices;

    std::vector<LitMaterial*> materials;

    SceneLoader() = default;
    ~SceneLoader();

    Mesh* findMeshByName(const std::string& name);
    Model* findModelByName(const std::string& name);
    Node* findNodeByName(const std::string& name);

    void loadScene(const std::string& filename, Scene& scene, PerspectiveCamera& camera);
    void clearScene(Scene& scene, PerspectiveCamera& camera);

private:
    void parse(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseSkybox(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseMaterial(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseMaterials(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseModel(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseModels(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseMesh(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseMeshes(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseNode(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseNodes(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseCamera(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseAmbientLight(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseDirectionalLight(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parsePointLight(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parsePointLights(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseAnimation(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
    void parseAnimations(const nlohmann::json& j, Scene& scene, PerspectiveCamera& camera);
};

} // namespace quasar

#endif // SCENE_LOADER_H
