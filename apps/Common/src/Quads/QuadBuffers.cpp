#include <spdlog/spdlog.h>

#include <Quads/QuadBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(size_t maxProxies)
    : maxProxies(maxProxies)
    , numProxies(maxProxies)
    , normalSphericalsBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , depthsBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(float),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , metadatasBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
#if defined(HAS_CUDA)
    , cudaBufferNormalSphericals(normalSphericalsBuffer)
    , cudaBufferDepths(depthsBuffer)
    , cudaBuffermetadatas(metadatasBuffer)
#endif
    , maxDataSize(sizeof(uint) + maxProxies * sizeof(QuadMapDataPacked))
{}

void QuadBuffers::resize(size_t numProxies) {
    this->numProxies = numProxies;
}

#ifdef GL_CORE
size_t QuadBuffers::mapToCPU(std::vector<char>& outputData) {
    outputData.resize(maxDataSize);
    size_t bufferOffset = 0;

    std::memcpy(outputData.data(), &numProxies, sizeof(uint));
    bufferOffset += sizeof(uint);

#if defined(HAS_CUDA)
    void* cudaPtr;

    cudaPtr = cudaBufferNormalSphericals.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(outputData.data() + bufferOffset, cudaPtr, numProxies * sizeof(uint), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(uint);

    cudaPtr = cudaBufferDepths.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(outputData.data() + bufferOffset, cudaPtr, numProxies * sizeof(float), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(float);

    cudaPtr = cudaBuffermetadatas.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(outputData.data() + bufferOffset, cudaPtr, numProxies * sizeof(uint), cudaMemcpyDeviceToHost));
    bufferOffset += numProxies * sizeof(uint);
#else
    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.getSubData(0, numProxies, outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint);

    depthsBuffer.bind();
    depthsBuffer.getSubData(0, numProxies, outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(float);

    metadatasBuffer.bind();
    metadatasBuffer.getSubData(0, numProxies, outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint);
#endif

    // Resize output
    outputData.resize(bufferOffset);
    return bufferOffset;
}
#endif

size_t QuadBuffers::unmapFromCPU(const std::vector<char>& inputData) {
    size_t bufferOffset = 0;

    numProxies = *reinterpret_cast<const uint*>(inputData.data());
    bufferOffset += sizeof(uint);

#if defined(HAS_CUDA)
    void* cudaPtr;

    cudaPtr = cudaBufferNormalSphericals.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(cudaPtr, inputData.data() + bufferOffset, numProxies * sizeof(uint), cudaMemcpyHostToDevice));
    bufferOffset += numProxies * sizeof(uint);

    cudaPtr = cudaBufferDepths.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(cudaPtr, inputData.data() + bufferOffset, numProxies * sizeof(float), cudaMemcpyHostToDevice));
    bufferOffset += numProxies * sizeof(float);

    cudaPtr = cudaBuffermetadatas.getPtr();
    CHECK_CUDA_ERROR(cudaMemcpy(cudaPtr, inputData.data() + bufferOffset, numProxies * sizeof(uint), cudaMemcpyHostToDevice));
    bufferOffset += numProxies * sizeof(uint);
#else
    auto normalSphericalsPtr = reinterpret_cast<const uint*>(inputData.data() + bufferOffset);
    normalSphericalsBuffer.bind();
    normalSphericalsBuffer.setData(numProxies, normalSphericalsPtr);
    bufferOffset += numProxies * sizeof(uint);

    auto depthsPtr = reinterpret_cast<const float*>(inputData.data() + bufferOffset);
    depthsBuffer.bind();
    depthsBuffer.setData(numProxies, depthsPtr);
    bufferOffset += numProxies * sizeof(float);

    auto metadatasPtr = reinterpret_cast<const uint*>(inputData.data() + bufferOffset);
    metadatasBuffer.bind();
    metadatasBuffer.setData(numProxies, metadatasPtr);
    bufferOffset += numProxies * sizeof(uint);
#endif

    // Set new number of proxies
    resize(numProxies);
    return numProxies;
}
