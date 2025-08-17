#include <spdlog/spdlog.h>
#include <Quads/DepthOffsets.h>

using namespace quasar;

static inline size_t bytesPerRow(unsigned w) {
    return static_cast<size_t>(w) * 4 * sizeof(uint16_t); // RGBA16F => 4 channels * 2 bytes
}

DepthOffsets::DepthOffsets(const glm::uvec2& size)
    : size(size)
    , buffer({
        .width = size.x,
        .height = size.y,
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
    , data(size.x * size.y * 4 * sizeof(uint16_t))
{}

#if defined(HAS_CUDA)
size_t DepthOffsets::mapToCPU(std::vector<char>& outputData) {
    size_t rowBytes = bytesPerRow(size.x);
    size_t total = rowBytes * size.y;

    outputData.resize(total);
    cudaImage.copyArrayToHost(rowBytes, size.y, rowBytes, outputData.data());
    return total;
}
#endif

size_t DepthOffsets::unmapFromCPU(std::vector<char>& inputData) {
    data = inputData;

#if defined(HAS_CUDA)
    size_t rowBytes = bytesPerRow(size.x);
    cudaImage.copyHostToArray(rowBytes, size.y, rowBytes, data.data());
#else
    buffer.setData(size.x, size.y, data.data());
#endif
    return data.size();
}
