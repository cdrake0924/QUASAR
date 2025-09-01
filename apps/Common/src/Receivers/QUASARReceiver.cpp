#include <Receivers/QUASARReceiver.h>

namespace quasar {

QUASARReceiver::QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide)
    : quadSet(quadSet)
    , maxLayers(maxLayers)
    , remoteCamera(quadSet.getSize())
    , remoteCameraWideFov(quadSet.getSize())
    , frames(maxLayers)
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
{
    remoteCamera.setFovyDegrees(remoteFOV);

    // Camera has wider FOV
    remoteCameraWideFov.setFovyDegrees(remoteFOVWide);
    remoteCameraWideFov.setViewMatrix(remoteCamera.getViewMatrix());

    threadPool = std::make_unique<BS::thread_pool<>>(maxLayers);

    TextureDataCreateParams texParams = {
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    };
    colorTextures.reserve(maxLayers);
    meshes.reserve(maxLayers);
    for (int layer = 0; layer < maxLayers; layer++) {
        colorTextures.emplace_back(texParams);
        // First and last layer need a lot of quads, each subsequent one has less
        uint maxProxies = (layer == 0 || layer == maxLayers - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 4;
        meshes.emplace_back(quadSet, colorTextures[layer], maxProxies);
    }
}

QuadMesh& QUASARReceiver::getMesh(int layer) {
    return meshes[layer];
}

PerspectiveCamera& QUASARReceiver::getRemoteCamera() {
    return remoteCamera;
}

PerspectiveCamera& QUASARReceiver::getRemoteCameraWideFov() {
    return remoteCameraWideFov;
}

void QUASARReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };
    for (int layer = 0; layer < maxLayers; layer++) {
        // Load texture
        Path colorFileName = dataPath / ("color" + std::to_string(layer) + ".jpg");
        colorTextures[layer].loadFromFile(colorFileName, true, false);

        // Load quads and depth offsets from files and decompress (nonblocking)
        double startTime = timeutils::getTimeMicros();
        frames[layer].loadFromFiles(dataPath, layer);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        auto offsetsFuture = threadPool->submit_task([&]() {
            return frames[layer].decompressDepthOffsets(uncompressedOffsets);
        });
        auto quadsFuture = threadPool->submit_task([&]() {
            return frames[layer].decompressQuads(uncompressedQuads);
        });
        quadsFuture.get(); offsetsFuture.get();
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Copy data to GPU
        auto sizes = quadSet.loadFromMemory(uncompressedQuads, uncompressedOffsets);
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Update mesh
        const glm::vec2& gBufferSize = glm::vec2(colorTextures[layer].width, colorTextures[layer].height);
        auto& cameraToUse = getCameraToUse(layer);
        startTime = timeutils::getTimeMicros();
        meshes[layer].appendQuads(quadSet, gBufferSize);
        meshes[layer].createMeshFromProxies(quadSet, gBufferSize, cameraToUse);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[layer].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }
}

} // namespace quasar
