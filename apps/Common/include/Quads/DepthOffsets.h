#ifndef DEPTH_OFFSETS_H
#define DEPTH_OFFSETS_H

#include <Path.h>
#include <Texture.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthOffsets {
public:
    glm::uvec2 size;
    Texture buffer;

    DepthOffsets(const glm::uvec2& size);
    ~DepthOffsets() = default;

#if defined(HAS_CUDA)
    size_t mapToCPU(std::vector<char>& outputData);
    size_t saveToFile(const Path& filename);
#endif
    size_t unmapFromCPU(std::vector<char>& inputData);

private:
    std::vector<char> data;

#if defined(HAS_CUDA)
    CudaGLImage cudaImage;
#endif
};

} // namespace quasar

#endif // DEPTH_OFFSETS_H
