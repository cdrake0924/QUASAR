#ifndef POINT_SHADOW_MAP_MATERIAL_H
#define POINT_SHADOW_MAP_MATERIAL_H

#include <Materials/ShadowMapMaterial.h>

namespace quasar {

class PointShadowMapMaterial : public ShadowMapMaterial {
public:
    PointShadowMapMaterial();
    ~PointShadowMapMaterial() = default;

    void bind() const override {
        getShader()->bind();
    }

    std::shared_ptr<Shader> getShader() const override {
        return shader;
    }

    static std::shared_ptr<Shader> shader;
};

} // namespace quasar

#endif // POINT_SHADOW_MAP_MATERIAL_H
