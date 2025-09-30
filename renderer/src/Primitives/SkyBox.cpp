#include <Primitives/SkyBox.h>

using namespace quasar;

SkyBox::SkyBox(const CubeMapCreateParams& params)
    : CubeMap(params)
    , irradianceCubeMap({
        .width = 32,
        .height = 32,
        .type = CubeMapType::STANDARD,
    })
    , prefilterCubeMap({
        .width = 128,
        .height = 128,
        .type = CubeMapType::PREFILTER,
    })
    , captureRenderTarget({
        .width = params.width,
        .height = params.height,
        .internalFormat = GL_RGB16F,
        .format = GL_RGB,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    })
    , captureRenderBuffer({
        .width = params.width,
        .height = params.height,
    })
    , brdfLUT({
        .internalFormat = GL_RG16F,
        .format = GL_RG,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_LINEAR,
        .magFilter = GL_LINEAR,
    })
    , equirectToCubeMapShader({
        .vertexCodeData = SHADER_BUILTIN_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_BUILTIN_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_EQUIRECTANGULAR2CUBEMAP_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_EQUIRECTANGULAR2CUBEMAP_FRAG_len,
    })
    , convolutionShader({
        .vertexCodeData = SHADER_BUILTIN_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_BUILTIN_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_IRRADIANCE_CONVOLUTION_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_IRRADIANCE_CONVOLUTION_FRAG_len,
    })
    , prefilterShader({
        .vertexCodeData = SHADER_BUILTIN_CUBEMAP_VERT,
        .vertexCodeSize = SHADER_BUILTIN_CUBEMAP_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_PREFILTER_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_PREFILTER_FRAG_len,
    })
    , brdfShader({
        .vertexCodeData = SHADER_BUILTIN_BRDF_VERT,
        .vertexCodeSize = SHADER_BUILTIN_BRDF_VERT_len,
        .fragmentCodeData = SHADER_BUILTIN_BRDF_FRAG,
        .fragmentCodeSize = SHADER_BUILTIN_BRDF_FRAG_len,
    })
{
    setupIBL();
}

SkyBox::SkyBox(const CubeMapCreateParams& params, const Texture& hdrTexture)
    : SkyBox(params)
{
    equirectToCubeMap(hdrTexture);
    setupIBL();
}

void SkyBox::equirectToCubeMap(const Texture& hdrTexture) {
    captureRenderTarget.bind();
    captureRenderTarget.resize(width, height);
    loadFromEquirectTexture(equirectToCubeMapShader, hdrTexture);
    captureRenderTarget.unbind();
}

void SkyBox::setupIBL() {
    captureRenderTarget.resize(width, height);

    glDisable(GL_BLEND);

    // Create an irradiance cubemap, and rescale capture FBO to irradiance scale
    captureRenderTarget.bind();
    captureRenderBuffer.bind();
    captureRenderBuffer.resize(irradianceCubeMap.width, irradianceCubeMap.height);

    // Convolve the environment map into the irradiance map
    captureRenderTarget.bind();
    irradianceCubeMap.convolve(convolutionShader, *this);
    captureRenderTarget.unbind();

    // Create a prefilter cubemap, and rescale capture FBO to prefilter scale
    captureRenderTarget.bind();
    prefilterCubeMap.prefilter(prefilterShader, *this, captureRenderBuffer);
    captureRenderTarget.unbind();

    // Generate a 2D LUT from the BRDF equations used
    brdfLUT.resize(width, height);

    // Reuse capture FBO and RBO
    captureRenderTarget.bind();
    captureRenderBuffer.bind();
    captureRenderBuffer.resize(brdfLUT.width, brdfLUT.height);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUT.ID, 0);

    // Render BRDF LUT to FBO
    brdfShader.bind();
    glViewport(0, 0, brdfLUT.width, brdfLUT.height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    brdfFsQuad.draw();

    captureRenderTarget.unbind();
    captureRenderBuffer.unbind();

    glEnable(GL_BLEND);
}
