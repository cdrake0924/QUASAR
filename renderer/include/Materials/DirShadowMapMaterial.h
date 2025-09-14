#ifndef DIR_SHADOW_MAP_MATERIAL_H
#define DIR_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

namespace quasar {

class DirShadowMapMaterial : public ShadowMapMaterial {
public:
    DirShadowMapMaterial();
    ~DirShadowMapMaterial() = default;

    void bind() const override {
        getShader()->bind();
    }

    std::shared_ptr<Shader> getShader() const override {
        return shader;
    }

    static std::shared_ptr<Shader> shader;
};

} // namespace quasar

#endif // DIR_SHADOW_MAP_MATERIAL_H
