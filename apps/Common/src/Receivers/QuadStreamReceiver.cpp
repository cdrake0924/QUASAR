#include <Receivers/QuadStreamReceiver.h>

using namespace quasar;

QuadStreamReceiver::QuadStreamReceiver(QuadSet& quadSet, uint maxViews, float remoteFOV, float remoteFOVWide, float viewBoxSize)
    : quadSet(quadSet)
    , maxViews(maxViews)
    , frames(maxViews)
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
{
    remoteCameras.reserve(maxViews);
    for (int view = 0; view < maxViews; view++) {
        remoteCameras.emplace_back(quadSet.getSize());
        remoteCameras[view].setFovyDegrees(remoteFOV);
    }

    threadPool = std::make_unique<BS::thread_pool<>>(maxViews);

    PerspectiveCamera& remoteCameraCenter = remoteCameras[0];
    for (int view = 1; view < maxViews - 1; view++) {
        const glm::vec3& offset = offsets[view - 1];
        const glm::vec3& right = remoteCameraCenter.getRightVector();
        const glm::vec3& up = remoteCameraCenter.getUpVector();
        const glm::vec3& forward = remoteCameraCenter.getForwardVector();

        glm::vec3 worldOffset =
            right   * offset.x * viewBoxSize / 2.0f +
            up      * offset.y * viewBoxSize / 2.0f +
            forward * -offset.z * viewBoxSize / 2.0f;

        remoteCameras[view].setViewMatrix(remoteCameraCenter.getViewMatrix());
        remoteCameras[view].setPosition(remoteCameraCenter.getPosition() + worldOffset);
        remoteCameras[view].updateViewMatrix();
    }

    // Last camera has wider FOV
    remoteCameras[maxViews-1].setFovyDegrees(remoteFOVWide);
    remoteCameras[maxViews-1].setViewMatrix(remoteCameraCenter.getViewMatrix());

    TextureDataCreateParams texParams = {
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    };
    colorTextures.reserve(maxViews);
    meshes.reserve(maxViews);
    for (int view = 0; view < maxViews; view++) {
        colorTextures.emplace_back(texParams);
        // We can use less vertices and indicies for the additional views since they will be sparser
        uint maxProxies = (view == 0 || view == maxViews - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 4;
        meshes.emplace_back(quadSet, colorTextures[view], maxProxies);
    }
}

QuadMesh& QuadStreamReceiver::getMesh(int view) {
    return meshes[view];
}

PerspectiveCamera& QuadStreamReceiver::getRemoteCamera(int view) {
    return remoteCameras[view];
}

void QuadStreamReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };
    for (int view = 0; view < maxViews; view++) {
        // Load texture
        Path colorFileName = dataPath / ("color" + std::to_string(view) + ".jpg");
        colorTextures[view].loadFromFile(colorFileName, true, false);

        // Load quads and depth offsets from files and decompress (nonblocking)
        double startTime = timeutils::getTimeMicros();
        frames[view].loadFromFiles(dataPath, view);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        startTime = timeutils::getTimeMicros();
        auto offsetsFuture = threadPool->submit_task([&]() {
            return frames[view].decompressDepthOffsets(uncompressedOffsets);
        });
        auto quadsFuture = threadPool->submit_task([&]() {
            return frames[view].decompressQuads(uncompressedQuads);
        });
        quadsFuture.get(); offsetsFuture.get();
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Copy data to GPU
        auto sizes = quadSet.loadFromMemory(uncompressedQuads, uncompressedOffsets);
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Update mesh
        const glm::vec2& gBufferSize = glm::vec2(colorTextures[view].width, colorTextures[view].height);
        startTime = timeutils::getTimeMicros();
        meshes[view].appendQuads(quadSet, gBufferSize);
        meshes[view].createMeshFromProxies(quadSet, gBufferSize, remoteCameras[view]);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = meshes[view].getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }
}
