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
    : maxProxies(maxQuadsPerMesh)
    , currentQuadBuffers(maxProxies)
    , meshSizesBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(BufferSizes),
        .numElems = 1,
        .usage = GL_DYNAMIC_COPY,
    })
    , quadIndicesMap({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(uint),
        .numElems = quadSet.getSize().x * quadSet.getSize().y,
        .usage = GL_DYNAMIC_DRAW,
    })
    , quadCreatedBuffer({
        .target = GL_SHADER_STORAGE_BUFFER,
        .dataSize = sizeof(int),
        .numElems = maxProxies,
        .usage = GL_DYNAMIC_DRAW,
    })
    , fillQuadIndicesShader({
        .computeCodeData = SHADER_COMMON_FILL_QUAD_INDICES_COMP,
        .computeCodeSize = SHADER_COMMON_FILL_QUAD_INDICES_COMP_len,
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
    meshSizesBuffer.getData(&bufferSizes);
    meshSizesBuffer.unbind();
    return bufferSizes;
}

void QuadMesh::appendQuads(const QuadSet& quadSet, const glm::vec2& gBufferSize, bool isReferenceFrame) {
    double startTime = timeutils::getTimeMicros();

    if (isReferenceFrame) {
        currNumProxies = 0;
        prevNumProxies = 0;
    }
    spdlog::warn(isReferenceFrame ? "Resetting proxy counts for reference frame" : "Updating proxy counts");

    uint newNumProxies = quadSet.getNumProxies();
    currNumProxies += newNumProxies;

    uint maxInvocations = std::min(currNumProxies, MAX_QUADS_PER_MESH);
    spdlog::info("{}: Previous number of proxies: {}", ID, prevNumProxies);
    spdlog::info("{}: New number of proxies: {}", ID, newNumProxies);
    spdlog::info("{}: Current number of proxies: {}", ID, currNumProxies);
    spdlog::info("{}: Max invocations: {}", ID, maxInvocations);

    fillQuadIndicesShader.bind();
    {
        fillQuadIndicesShader.setBool("isReferenceFrame", isReferenceFrame);
        fillQuadIndicesShader.setVec2("gBufferSize", gBufferSize);
        fillQuadIndicesShader.setUint("prevNumProxies", prevNumProxies);
        fillQuadIndicesShader.setUint("maxInvocations", maxInvocations);
    }
    {
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 0, quadSet.quadBuffers.normalSphericalsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 1, quadSet.quadBuffers.depthsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 2, quadSet.quadBuffers.metadatasBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 3, currentQuadBuffers.normalSphericalsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 4, currentQuadBuffers.depthsBuffer);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 5, currentQuadBuffers.metadatasBuffer);

        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 6, quadIndicesMap);
        fillQuadIndicesShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, quadCreatedBuffer);
    }
    fillQuadIndicesShader.dispatch(((maxInvocations + 1) + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1, 1);
    fillQuadIndicesShader.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    prevNumProxies = currNumProxies;

    stats.timeToGatherQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void QuadMesh::createMeshFromProxies(const QuadSet& quadSet, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera) {
    double startTime = timeutils::getTimeMicros();

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

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 7, quadIndicesMap);

        createQuadMeshShader.setBuffer(GL_SHADER_STORAGE_BUFFER, 8, quadCreatedBuffer);

        createQuadMeshShader.setImageTexture(0, quadSet.depthOffsets.texture, 0, GL_FALSE, 0, GL_READ_ONLY, quadSet.depthOffsets.texture.internalFormat);
    }
    createQuadMeshShader.dispatch((quadSet.getSize().x + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP,
                                  (quadSet.getSize().y + THREADS_PER_LOCALGROUP - 1) / THREADS_PER_LOCALGROUP, 1);
    createQuadMeshShader.memoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);

    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}
