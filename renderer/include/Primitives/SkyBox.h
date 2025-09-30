#ifndef SKYBOX_H
#define SKYBOX_H

#include <CubeMap.h>
#include <Shaders/Shader.h>
#include <Primitives/FullScreenQuad.h>
#include <RenderTargets/RenderTarget.h>

namespace quasar {

class SkyBox : public CubeMap {
public:
    SkyBox(const CubeMapCreateParams& params);
    SkyBox(const CubeMapCreateParams& params, const Texture& hdrTexture);

    const CubeMap& getIrradianceCubeMap() const { return irradianceCubeMap; }
    const CubeMap& getPrefilterCubeMap() const { return prefilterCubeMap; }
    const Texture& getBRDFLUT() const { return brdfLUT; }

private:
    // BRDF shader
    Shader brdfShader;

    // Converts HDR equirectangular environment map to cubemap equivalent
    Shader equirectToCubeMapShader;

    // Solves diffuse integral by convolution to create an irradiance cubemap
    Shader convolutionShader;

    // Runs a quasi monte-carlo simulation on the environment lighting to create a prefilter cubemap
    Shader prefilterShader;

    // Capture FBO and RBO
    RenderTarget captureRenderTarget;
    Renderbuffer captureRenderBuffer;

    // Create an irradiance cubemap, and rescale capture FBO to irradiance scale
    CubeMap irradianceCubeMap;

    // Create a prefilter cubemap, and rescale capture FBO to prefilter scale
    CubeMap prefilterCubeMap;

    // Generate a 2D LUT from the BRDF equations used
    Texture brdfLUT;
    FullScreenQuad brdfFsQuad;

    void equirectToCubeMap(const Texture& hdrTexture);
    void setupIBL();
};

} // namespace quasar

#endif // SKYBOX_H
