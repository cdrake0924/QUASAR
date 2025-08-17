#include <spdlog/spdlog.h>

#include <Quads/DepthOffsets.h>

using namespace quasar;

static inline size_t bytesPerRow(unsigned w) {
    return static_cast<size_t>(w) * 4 * sizeof(uint16_t); // RGBA16F => 4 channels * 2 bytes
}

DepthOffsets::DepthOffsets(const glm::uvec2& textureSize)
    : textureSize(textureSize)
    , buffer({
        .width = textureSize.x,
        .height = textureSize.y,
        .internalFormat = GL_RGBA16F, // 4 offsets per subpixel corner
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
#if defined(HAS_CUDA)
    , cudaImage(buffer)
#endif
{}

#if defined(HAS_CUDA)
size_t DepthOffsets::mapToCPU(std::vector<char>& outputData) {
    size_t rowBytes = bytesPerRow(textureSize.x);
    size_t outputSize = rowBytes * textureSize.y;
    outputData.resize(outputSize);
    cudaImage.copyArrayToHost(rowBytes, textureSize.y, rowBytes, outputData.data());
    return outputData.size();
}
#endif

size_t DepthOffsets::unmapFromCPU(std::vector<char>& inputData) {
#if defined(HAS_CUDA)
    size_t rowBytes = bytesPerRow(textureSize.x);
    cudaImage.copyHostToArray(rowBytes, textureSize.y, rowBytes, inputData.data());
#else
    buffer.loadFromData(inputData.data());
#endif
    return inputData.size();
}
