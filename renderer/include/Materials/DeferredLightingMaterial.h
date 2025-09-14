#ifndef DEFERRED_LIGHTING_MATERIAL_H
#define DEFERRED_LIGHTING_MATERIAL_H

#include <RenderTargets/GBuffer.h>
#include <Materials/Material.h>

namespace quasar {

class DeferredLightingMaterial : public Material {
public:
    DeferredLightingMaterial();
    ~DeferredLightingMaterial() = default;

    void bindGBuffer(const GBuffer& FrameRenderTarget) const;
    void bindCamera(const Camera& camera) const;

    void bind() const override {
        getShader()->bind();
    }

    std::shared_ptr<Shader> getShader() const override {
        return shader;
    }

    uint getTextureCount() const override { return 7; }

    static std::shared_ptr<Shader> shader;
};

} // namespace quasar

#endif // DEFERRED_LIGHTING_MATERIAL_H
