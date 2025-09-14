#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/scene.h>

#include <Vertex.h>
#include <Shaders/Shader.h>
#include <Primitives/Mesh.h>
#include <Materials/LitMaterial.h>
#include <Primitives/Entity.h>

namespace quasar {

struct ModelCreateParams {
    bool flipTextures = false;
    bool gammaCorrected = false;
    float IBL = 1.0;
    const LitMaterial* material;
    std::string path;
};

class Model : public Entity {
public:
    std::vector<Mesh*> meshes;
    Node rootNode;

    std::string rootDirectory;

    bool flipTextures = false;
    bool gammaCorrected = false;
    float IBL = 1.0;

    const LitMaterial* material;

    bool isGLTF = false;

    Model(const ModelCreateParams& params);
    ~Model();

    virtual void bindMaterial(Scene& scene, Buffer& pointLightsUBO,
                              const Material* overrideMaterial = nullptr, const Texture* prevIDMap = nullptr) override;

    virtual RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model,
                             bool frustumCull = true, const Material* overrideMaterial = nullptr) override;
    virtual RenderStats draw(GLenum primitiveType, const Camera& camera, const glm::mat4& model,
                             const BoundingSphere& boundingSphere, const Material* overrideMaterial = nullptr) override;
    virtual void updateAnimations(float dt) override;

    Node* findNodeByName(const std::string& name);

private:
    const aiScene* scene;
    std::unordered_map<std::string, Texture*> texturesLoaded;

    void loadFromFile(const ModelCreateParams& params);
    void processNode(aiNode* aiNode, const aiScene* scene, Node* node, const LitMaterial* material);
    Mesh* processMesh(aiMesh* mesh, const aiScene* scene, const LitMaterial* material);
    void processAnimations(const aiScene* scene);
    void processMaterial(aiMaterial const* aiMat, LitMaterialCreateParams& materialParams);
    Texture* loadMaterialTexture(aiMaterial const* aiMat, aiString aiTexturePath, bool shouldGammaCorrect = false);
    int32_t getEmbeddedTextureId(const aiString& path);

    RenderStats drawNode(const Node* node,
                         GLenum primitiveType, const Camera& camera,
                         const glm::mat4& parentTransform, const glm::mat4& model,
                         bool frustumCull = true, const Material* overrideMaterial = nullptr);
    RenderStats drawNode(const Node* node,
                         GLenum primitiveType, const Camera& camera,
                         const glm::mat4& parentTransform, const glm::mat4& model,
                         const BoundingSphere& boundingSphere, const Material* overrideMaterial = nullptr);
};

} // namespace quasar

#endif // MODEL_H
