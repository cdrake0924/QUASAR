#ifndef TONE_MAP_EFFECT_H
#define TONE_MAP_EFFECT_H

#include <Shaders/TonemapShader.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class Tonemapper : public PostProcessingEffect {
public:
    Tonemapper(bool tonemap = true) {
        enableTonemapping(tonemap);
        setExposure(1.0f);
    }

    void enableTonemapping(bool enable) {
        shader.bind();
        shader.setBool("tonemap", enable);
    }

    void setExposure(float exposure) {
        shader.bind();
        shader.setFloat("exposure", exposure);
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

    void setUniforms(FrameRenderTarget& rt) {
        shader.bind();
        shader.setTexture("screenColor", rt.colorTexture, 0);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt, bool setUniforms) {
        if (setUniforms) renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    TonemapShader shader;
};

} // namespace quasar

#endif // TONE_MAP_EFFECT_H
