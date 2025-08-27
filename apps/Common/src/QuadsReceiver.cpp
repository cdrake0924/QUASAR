#include <cstring>
#include <stdexcept>

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <QuadsReceiver.h>

namespace quasar {

QuadsReceiver::QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& streamerURL)
    : quadSet(quadSet)
    , streamerURL(streamerURL)
    , remoteCamera(quadSet.getSize())
    , colorTexture({
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , referenceFrameMesh(quadSet, colorTexture)
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, colorTexture, MAX_QUADS_PER_MESH / 4)
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedQuadsUpdated(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedQuadsRevealed(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , uncompressedOffsetsUpdated(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , uncompressedOffsetsRevealed(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , DataReceiverTCP(streamerURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    if (!streamerURL.empty()) {
        spdlog::info("Created QuadsReceiver that recvs from URL: {}", streamerURL);
    }
}

QuadMesh& QuadsReceiver::getReferenceMesh() {
    return referenceFrameMesh;
}

QuadMesh& QuadsReceiver::getResidualMesh() {
    return residualFrameMesh;
}

void QuadsReceiver::copyPoseToCamera(PerspectiveCamera& camera) {
    camera.setProjectionMatrix(cameraPose.mono.proj);
    camera.setViewMatrix(cameraPose.mono.view);
}

void QuadsReceiver::onDataReceived(const std::vector<char>& data) {
    std::lock_guard<std::mutex> lock(m);
    frames.push_back(std::move(data));
}

FrameType QuadsReceiver::recvData() {
    FrameType frameType = FrameType::NONE;
    std::lock_guard<std::mutex> lock(m);
    if (!frames.empty()) {
        auto data = frames.front();
        frames.pop_front();
        frameType = loadFromMemory(data);
    }
    return frameType;
}

FrameType QuadsReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };
    double startTime = timeutils::getTimeMicros();

    std::string cameraFileName = dataPath / "camera.bin";
    cameraPose.loadFromFile(cameraFileName);
    copyPoseToCamera(remoteCamera);

    std::string colorFileName = dataPath / "color.jpg";
    colorTexture.loadFromFile(colorFileName, true, false);

    referenceFrame.loadFromFiles(dataPath);
    stats.loadTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    updateGeometry(FrameType::REFERENCE);
    return FrameType::REFERENCE;
}

FrameType QuadsReceiver::loadFromMemory(const std::vector<char>& inputData) {
    stats = { 0 };
    double startTime = timeutils::getTimeMicros();

    const char* ptr = inputData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    // Sanity check
    if (inputData.size() < sizeof(Header) + header.cameraSize + header.colorSize + header.geometrySize) {
        throw std::runtime_error("Input data size " +
                                  std::to_string(inputData.size()) +
                                  " is smaller than expected from header " +
                                  std::to_string(header.colorSize + header.geometrySize));
    }

    spdlog::debug("Reading camera size: {}", header.cameraSize);
    spdlog::debug("Reading color size: {}", header.colorSize);
    spdlog::debug("Reading geometry size: {}", header.geometrySize);

    // Read camera data
    cameraPose.loadFromMemory(ptr, header.cameraSize);
    copyPoseToCamera(remoteCamera);
    ptr += header.cameraSize;

    // Read color data
    colorData.resize(header.colorSize);
    std::memcpy(colorData.data(), ptr, header.colorSize);
    ptr += header.colorSize;

    // Read geometry data
    geometryData.resize(header.geometrySize);
    std::memcpy(geometryData.data(), ptr, header.geometrySize);
    ptr += header.geometrySize;

    // Run async
    auto future = std::async(std::launch::async, [this]() {
        int width, height, channels;
        FileIO::flipVerticallyOnLoad(true);
        unsigned char* data = FileIO::loadImageFromMemory(colorData.data(), colorData.size(), &width, &height, &channels);
        return std::make_tuple(data, width, height, channels);
    });

    // Load QuadFrame
    if (header.frameType == FrameType::REFERENCE) {
        referenceFrame.loadFromMemory(geometryData);
    }
    else {
        residualFrame.loadFromMemory(geometryData);
    }
    stats.loadTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    updateGeometry(header.frameType);

    // Wait for async to finish
    auto [data, width, height, channels] = future.get();
    colorTexture.resize(width, height);
    colorTexture.loadFromData(data);
    FileIO::freeImage(data);

    return header.frameType;
}

void QuadsReceiver::updateGeometry(FrameType frameType) {
    const glm::vec2& gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);
    if (frameType == FrameType::REFERENCE) {
        double startTime = timeutils::getTimeMicros();
        // Decompress proxies (asynchronous)
        auto offsetsFuture = referenceFrame.decompressDepthOffsets(uncompressedOffsets);
        auto quadsFuture = referenceFrame.decompressQuads(uncompressedQuads);
        quadsFuture.get(); offsetsFuture.get();
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Transfer updated proxies to GPU for reconstruction
        auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
        referenceFrame.numQuads = sizes.numQuads;
        referenceFrame.numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct mesh using proxies
        startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }
    else {
        double startTime = timeutils::getTimeMicros();
        // Decompress proxies (asynchronous)
        auto offsetsUpdatedFuture = residualFrame.decompressUpdatedDepthOffsets(uncompressedOffsetsUpdated);
        auto offsetsRevealedFuture = residualFrame.decompressRevealedDepthOffsets(uncompressedOffsetsRevealed);
        auto quadsUpdatedFuture = residualFrame.decompressUpdatedQuads(uncompressedQuadsUpdated);
        auto quadsRevealedFuture = residualFrame.decompressRevealedQuads(uncompressedQuadsRevealed);
        quadsUpdatedFuture.get(); offsetsUpdatedFuture.get();
        stats.timeToDecompressMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Transfer updated proxies to GPU for reconstruction
        auto sizesUpdated = quadSet.copyFromCPU(uncompressedQuadsUpdated, uncompressedOffsetsUpdated);
        residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
        residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, update reference frame mesh using proxies
        startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize, false /* is not reference frame; append */);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // This will also wait for the GPU to finish
        auto refMeshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles += refMeshBufferSizes.numIndices / 3;

        quadsRevealedFuture.get(); offsetsRevealedFuture.get();

        // Transfer revealed proxies to GPU for reconstruction
        auto sizesRevealed = quadSet.copyFromCPU(uncompressedQuadsRevealed, uncompressedOffsetsRevealed);
        residualFrame.numQuadsRevealed = sizesRevealed.numQuads;
        residualFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct revealed mesh using proxies
        startTime = timeutils::getTimeMicros();
        residualFrameMesh.appendQuads(quadSet, gBufferSize);
        residualFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto resMeshBufferSizes = residualFrameMesh.getBufferSizes();
        stats.totalTriangles += resMeshBufferSizes.numIndices / 3;
        stats.sizes += sizesUpdated;
        stats.sizes += sizesRevealed;
    }

    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
}

} // namespace quasar
