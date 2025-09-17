#ifndef QUAD_BUFFERS_H
#define QUAD_BUFFERS_H

#include <glm/glm.hpp>

#include <Buffer.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLBuffer.h>
#endif

namespace quasar {

struct QuadMapData {
    glm::vec3 normal;
    float depth;
    glm::ivec2 offset;
    uint32_t size;
    bool flattened;
};

struct QuadMapDataPacked {
    // normal is converted into spherical coordinates quantized to 8 bits (16 bits).
    // depth is quantized to 16 bits.
    // (normal theta << 24) | (normal phi << 16) | (depth). 16 + 16 bits = 32 bits used.
    uint32_t normalSphericalDepth;
    // (offset.x << 20) | (offset.y << 8) | (size << 1) | (flattened). 12 + 12 + 6 + 1 bits = 32 bits used.
    uint32_t metadata;
}; // 64 bits total

class QuadBuffers {
public:
    size_t maxProxies;
    size_t numProxies;

    Buffer normalSphericalDepthBuffer;
    Buffer metadatasBuffer;

    QuadBuffers(size_t maxProxies);
    ~QuadBuffers() = default;

    void resize(size_t newNumProxies);

#ifdef GL_CORE
    size_t writeToMemory(std::vector<char>& outputData, bool applyDeltaEncoding = true);
#endif
    size_t loadFromMemory(std::vector<char>& inputData, bool applyDeltaEncoding = true);

private:
#if defined(HAS_CUDA)
    CudaGLBuffer cudaBufferNormalSphericalDepth;
    CudaGLBuffer cudaBufferMetadatas;
#endif
};

} // namespace quasar

#endif // QUAD_BUFFERS_H
