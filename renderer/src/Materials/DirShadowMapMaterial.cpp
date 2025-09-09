#include <Materials/DirShadowMapMaterial.h>

using namespace quasar;

std::shared_ptr<Shader> DirShadowMapMaterial::shader = nullptr;

DirShadowMapMaterial::DirShadowMapMaterial() {
    if (shader == nullptr) {
        ShaderDataCreateParams dirShadowMapParams{
            .vertexCodeData = SHADER_BUILTIN_DIRSHADOW_VERT,
            .vertexCodeSize = SHADER_BUILTIN_DIRSHADOW_VERT_len,
            .fragmentCodeData = SHADER_BUILTIN_DIRSHADOW_FRAG,
            .fragmentCodeSize = SHADER_BUILTIN_DIRSHADOW_FRAG_len,
        };
        shader = std::make_shared<Shader>(dirShadowMapParams);
    }
}
