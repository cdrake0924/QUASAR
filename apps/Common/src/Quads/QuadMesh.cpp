#include <algorithm>
#include <Quads/QuadMesh.h>
#include <Utils/TimeUtils.h>
#include <shaders_common.h>

#ifndef __ANDROID__
#define THREADS_PER_LOCALGROUP 32
#else
#define THREADS_PER_LOCALGROUP 16
#endif

using namespace quasar;

QuadMesh::QuadMesh(const QuadSet& quadSet, Texture& colorTexture, uint maxQuadsPerMesh)
    : currentQuadBuffers(maxQuadsPerMesh)
    , meshSizesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BufferSizes),
        .numElems = 1,
        .usage = GL_DYNAMIC_COPY,
    })
    , quadIndexMap({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint),
        .numElems = quadSet.getSize().x * quadSet.getSize().y,
        .usage = GL_DYNAMIC_DRAW,
    })
    , quadCreatedBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(int),
        .numElems = maxQuadsPerMesh,
        .usage = GL_DYNAMIC_DRAW,
    })
    , appendQuadsShader({
        .computeCodeData = SHADER_COMMON_APPEND_QUADS_COMP,
        .computeCodeSize = SHADER_COMMON_APPEND_QUADS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , createQuadMeshShader({
        .computeCodeData = SHADER_COMMON_MESH_FROM_QUADS_COMP,
        .computeCodeSize = SHADER_COMMON_MESH_FROM_QUADS_COMP_len,
        .defines = {
            "#define THREADS_PER_LOCALGROUP " + std::to_string(THREADS_PER_LOCALGROUP)
        }
    })
    , Mesh({
        .maxVertices = maxQuadsPerMesh * NUM_SUB_QUADS * VERTICES_IN_A_QUAD,
        .maxIndices = maxQuadsPerMesh * NUM_SUB_QUADS * INDICES_IN_A_QUAD,
        .vertexSize = sizeof(QuadVertex),
        .attributes = QuadVertex::getVertexInputAttributes(),
        .material = new QuadMaterial({ .baseColorTexture = &colorTexture }),
        .usage = GL_DYNAMIC_DRAW,
        .indirectDraw = true
    })
{}

QuadMesh::BufferSizes QuadMesh::getBufferSizes() const {
    BufferSizes bufferSizes;
    meshSizesBuffer.bind();
    void* ptr = meshSizesBuffer.mapToCPU(GL_MAP_READ_BIT);
    if (ptr) {
        std::memcpy(&bufferSizes, ptr, sizeof(BufferSizes));
        meshSizesBuffer.unmapFromCPU();
    }
    else {
        spdlog::warn("Failed to map meshSizesBuffer. Copying using getData");
        meshSizesBuffer.getData(&bufferSizes);
    }
    return bufferSizes;
}

void QuadMesh::appendQuads(const QuadSet& quadSet, const glm::vec2& gBufferSize, bool isFullFrame) {
    double startTime = timeutils::getTimeMicros();

    if (isFullFrame) {
        currNumProxies = 0;
    }

    uint incomingNumProxies = quadSet.getNumProxies();
    uint newCurrNumProxies = currNumProxies + incomingNumProxies;
    if (newCurrNumProxies > MAX_QUADS_PER_MESH) {
        spdlog::warn("Max proxies reached! Clamping to {} proxies.", MAX_QUADS_PER_MESH);
        newCurrNumProxies = MAX_QUADS_PER_MESH;
    }

    appendQuadsShader.bind();
    {
        appendQuadsShader.setVec2("gBufferSize", gBufferSize);
        appendQuadsShader.setUint("currNumProxies", currNumProxies);
        appendQuadsShader.setUint("incomingNumProxies", incomingNumProxies);
    }
    {
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, quadSet.quadBuffers.normalSphericalsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadSet.quadBuffers.depthsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadSet.quadBuffers.metadatasBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currentQuadBuffers.normalSphericalsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currentQuadBuffers.depthsBuffer);
        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.metadatasBuffer);

        appendQuadsShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, quadIndexMap);
    }
    appendQuadsShader.dispatch(((incomingNumProxies + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    appendQuadsShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Set new current proxy count
    currNumProxies = newCurrNumProxies;

    stats.timeToGatherQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void QuadMesh::createMeshFromProxies(const QuadSet& quadSet, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera) {
    double startTime = timeutils::getTimeMicros();

    // Clear the quadCreatedBuffer
    quadCreatedBuffer.bind();
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I,  GL_RED_INTEGER, GL_INT, nullptr);

    createQuadMeshShader.bind();
    {
        createQuadMeshShader.setVec2("gBufferSize", gBufferSize);
    }
    {
        createQuadMeshShader.setMat4("view", remoteCamera.getViewMatrix());
        createQuadMeshShader.setMat4("projection", remoteCamera.getProjectionMatrix());
        createQuadMeshShader.setMat4("viewInverse", remoteCamera.getViewMatrixInverse());
        createQuadMeshShader.setMat4("projectionInverse", remoteCamera.getProjectionMatrixInverse());
        createQuadMeshShader.setFloat("near", remoteCamera.getNear());
        createQuadMeshShader.setFloat("far", remoteCamera.getFar());
    }
    {
        createQuadMeshShader.setUint("currNumProxies", currNumProxies);
    }
    {
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, meshSizesBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, vertexBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, indexBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, indirectBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currentQuadBuffers.normalSphericalsBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.depthsBuffer);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, currentQuadBuffers.metadatasBuffer);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, quadIndexMap);
        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadCreatedBuffer);

        createQuadMeshShader.setImageTexture(0, quadSet.depthOffsets.texture, 0, GL_FALSE, 0, GL_READ_ONLY, quadSet.depthOffsets.texture.internalFormat);
    }
    createQuadMeshShader.dispatch((quadSet.getSize().x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  (quadSet.getSize().y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createQuadMeshShader.memoryBarrier(GL_SHADER_STORAGE_BUFFER |
                                       GL_COMMAND_BARRIER_BIT |
                                       GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}
