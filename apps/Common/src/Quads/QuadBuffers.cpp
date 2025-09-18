#include <spdlog/spdlog.h>

#include <Quads/QuadBuffers.h>

using namespace quasar;

QuadBuffers::QuadBuffers(size_t maxProxies)
    : maxProxies(maxProxies)
    , numProxies(maxProxies)
    , normalSphericalDepthBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .maxElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
    , metadatasBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint32_t),
        .numElems = maxProxies,
        .maxElems = maxProxies,
        .usage = GL_DYNAMIC_COPY,
    })
#if defined(HAS_CUDA)
    , cudaBufferNormalSphericalDepth(normalSphericalDepthBuffer)
    , cudaBufferMetadatas(metadatasBuffer)
#endif
{}

void QuadBuffers::resize(size_t newNumProxies) {
    numProxies = newNumProxies;
    spdlog::debug("Resized QuadBuffers to {} proxies", newNumProxies);
}

#ifdef GL_CORE
size_t QuadBuffers::writeToMemory(std::vector<char>& outputData, bool applyDeltaEncoding) {
    outputData.resize(sizeof(uint32_t) + maxProxies * sizeof(QuadMapDataPacked));
    size_t bufferOffset = 0;

    std::memcpy(outputData.data(), &numProxies, sizeof(uint32_t));
    bufferOffset += sizeof(uint32_t);

    uint32_t* normalSphericalDepth = reinterpret_cast<uint32_t*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint32_t);

    uint32_t* metadatas = reinterpret_cast<uint32_t*>(outputData.data() + bufferOffset);
    bufferOffset += numProxies * sizeof(uint32_t);

#if defined(HAS_CUDA)
    CudaGLBuffer::registerHostBuffer(outputData.data(), outputData.size());

    cudaBufferNormalSphericalDepth.copyToHostAsync(normalSphericalDepth, numProxies * sizeof(uint32_t));
    cudaBufferMetadatas.copyToHostAsync(metadatas, numProxies * sizeof(uint32_t));

    cudaBufferNormalSphericalDepth.synchronize();
    CudaGLBuffer::unregisterHostBuffer(outputData.data());
#else
    void* ptr;

    normalSphericalsBuffer.bind();
    ptr = normalSphericalsBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(normalSphericalDepth, ptr, numProxies * sizeof(uint32_t));
        normalSphericalsBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalSphericalsBuffer. Copying using getData");
        normalSphericalsBuffer.getData(outputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint32_t);

    metadatasBuffer.bind();
    ptr = metadatasBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(metadatas, ptr, numProxies * sizeof(uint32_t));
        metadatasBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map metadatasBuffer. Copying using getData");
        metadatasBuffer.getData(outputData.data() + bufferOffset);
    }
    bufferOffset += numProxies * sizeof(uint32_t);
#endif

    if (applyDeltaEncoding) {
        // Apply delta encoding
        for (int i = numProxies - 1; i > 0; i--) {
            normalSphericalDepth[i] -= normalSphericalDepth[i - 1];
            metadatas[i] -= metadatas[i - 1];
        }
    }

    // Resize output
    outputData.resize(bufferOffset);
    return bufferOffset;
}
#endif

size_t QuadBuffers::loadFromMemory(std::vector<char>& inputData, bool applyDeltaEncoding) {
    if (inputData.size() < sizeof(uint32_t)) {
        return 0;
    }

    size_t bufferOffset = 0;
    void* ptr;

    uint32_t newNumProxies = *reinterpret_cast<const uint32_t*>(inputData.data());
    bufferOffset += sizeof(uint32_t);

    uint32_t* normalSphericalDepth = reinterpret_cast<uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(uint32_t);

    uint32_t* metadatas = reinterpret_cast<uint32_t*>(inputData.data() + bufferOffset);
    bufferOffset += newNumProxies * sizeof(uint32_t);

    if (applyDeltaEncoding) {
        // Decode delta encoding
        for (int i = 1; i < newNumProxies; i++) {
            normalSphericalDepth[i] += normalSphericalDepth[i - 1];
            metadatas[i] += metadatas[i - 1];
        }
    }

    normalSphericalDepthBuffer.bind();
    ptr = normalSphericalDepthBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, normalSphericalDepth, newNumProxies * sizeof(uint32_t));
        normalSphericalDepthBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map normalSphericalDepthBuffer. Copying using setData");
        normalSphericalDepthBuffer.setData(newNumProxies, reinterpret_cast<char*>(normalSphericalDepth));
    }

    metadatasBuffer.bind();
    ptr = metadatasBuffer.mapToCPU(GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    if (ptr) {
        std::memcpy(ptr, metadatas, newNumProxies * sizeof(uint32_t));
        metadatasBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map metadatasBuffer. Copying using setData");
        metadatasBuffer.setData(newNumProxies, reinterpret_cast<char*>(metadatas));
    }

    // Set new number of proxies
    resize(newNumProxies);
    return bufferOffset;
}
