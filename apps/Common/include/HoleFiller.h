#ifndef HOLE_FILLER_H
#define HOLE_FILLER_H

#include <PostProcessing/PostProcessingEffect.h>

namespace quasar {

class HoleFiller : public PostProcessingEffect {
public:
    HoleFiller()
        : shader({
            .vertexCodeData = SHADER_BUILTIN_POSTPROCESS_VERT,
            .vertexCodeSize = SHADER_BUILTIN_POSTPROCESS_VERT_len,
            .fragmentCodeData = SHADER_COMMON_HOLE_FILLER_FRAG,
            .fragmentCodeSize = SHADER_COMMON_HOLE_FILLER_FRAG_len,
        })
    {}

    void setDepthThreshold(float depthThreshold) {
        shader.bind();
        shader.setFloat("depthThreshold", depthThreshold);
    }

    void enableToneMapping(bool enable) {
        shader.bind();
        shader.setBool("toneMap", enable);
    }

    RenderStats drawToScreen(OpenGLRenderer& renderer) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToScreen(shader);
    }

    RenderStats drawToRenderTarget(OpenGLRenderer& renderer, RenderTargetBase& rt) override {
        renderer.setScreenShaderUniforms(shader);
        return renderer.drawToRenderTarget(shader, rt);
    }

private:
    Shader shader;
};

} // namespace quasar

#endif // HOLE_FILLER_H
