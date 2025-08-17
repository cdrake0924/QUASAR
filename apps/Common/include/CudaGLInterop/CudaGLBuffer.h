#ifndef CUDA_BUFFER_H
#define CUDA_BUFFER_H

#if defined(HAS_CUDA)

#include <Buffer.h>

#include <cuda_gl_interop.h>
#include <CudaGLInterop/CudaUtils.h>

namespace quasar {

class CudaGLBuffer {
public:
    CudaGLBuffer() = default;
    CudaGLBuffer(Buffer& buffer, cudaStream_t stream = nullptr)
        : buffer(&buffer)
        , stream(stream)
    {
        cudautils::checkCudaDevice();
        registerBuffer(buffer);
    }

    ~CudaGLBuffer() {
        if (ownsStream && stream) cudaStreamSynchronize(stream);
        if (cudaResource) {
            CHECK_CUDA_ERROR(cudaGraphicsUnregisterResource(cudaResource));
        }
        if (ownsStream && stream) cudaStreamDestroy(stream);
    }

    void registerBuffer(Buffer& buffer) {
        this->buffer = &buffer;
        CHECK_CUDA_ERROR(cudaGraphicsGLRegisterBuffer(
            &cudaResource,
            buffer,
            cudaGraphicsRegisterFlagsNone));
    }

    void ensureStream() {
        if (!stream) {
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            ownsStream = true;
        }
    }

    void map() {
        ensureStream();
        CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource, stream));
    }

    void unmap() {
        CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, stream));
    }

    void* getPtr() {
        void* cudaPtr; size_t size;
        map();
        CHECK_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size, cudaResource));
        unmap();
        return cudaPtr;
    }

    cudaStream_t getStream() const { return stream; }
    void setStream(cudaStream_t s) { stream = s; ownsStream = false; }

private:
    bool ownsStream = false;

    const Buffer* buffer = nullptr;

    cudaGraphicsResource* cudaResource = nullptr;
    cudaStream_t stream = nullptr;
};

} // namespace quasar

#endif // defined(HAS_CUDA)

#endif // CUDA_BUFFER_H
