#include <cstring>
#include <stdexcept>

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <QuadsReceiver.h>

using namespace quasar;

QuadsReceiver::QuadsReceiver(QuadSet& quadSet, const std::string& streamerURL)
    : quadSet(quadSet)
    , streamerURL(streamerURL)
    , remoteCamera(quadSet.getSize())
    , referenceColorTexture({
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , residualColorTexture({
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_REPEAT,
        .wrapT = GL_REPEAT,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , referenceFrameMesh(quadSet, referenceColorTexture)
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, residualColorTexture, MAX_QUADS_PER_MESH / 4)
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedQuadsUpdated(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedQuadsRevealed(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , uncompressedOffsetsUpdated(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , uncompressedOffsetsRevealed(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , DataReceiverTCP(streamerURL)
{
    if (!streamerURL.empty()) {
        spdlog::info("Created QuadsReceiver that recvs from URL: {}", streamerURL);
    }
}

QuadsReceiver::QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& streamerURL)
    : QuadsReceiver(quadSet, streamerURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
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

    Path cameraFileName = dataPath / "camera.bin";
    cameraPose.loadFromFile(cameraFileName);
    copyPoseToCamera(remoteCamera);

    Path colorFileNameRef = dataPath / "color.jpg";
    referenceColorTexture.loadFromFile(colorFileNameRef, true, false);

    referenceFrame.loadFromFiles(dataPath);
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    updateGeometry(FrameType::REFERENCE);

    Path cameraFileNamePrev = dataPath / "camera_prev.bin";
    cameraPose.loadFromFile(cameraFileNamePrev);
    copyPoseToCamera(remoteCameraPrev);

    Path colorFileNameRes = dataPath / "color_residual.jpg";
    residualColorTexture.loadFromFile(colorFileNameRes, true, false);

    residualFrame.loadFromFiles(dataPath);

    updateGeometry(FrameType::RESIDUAL);
    return FrameType::RESIDUAL;
}

FrameType QuadsReceiver::loadFromMemory(const std::vector<char>& inputData) {
    stats = { 0 };
    double startTime = timeutils::getTimeMicros();

    const char* ptr = inputData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    // Sanity check
    size_t expectedSize = sizeof(Header) +
                          header.cameraSize +
                          header.referenceColorSize + header.residualColorSize +
                          header.geometrySize;
    if (inputData.size() < expectedSize) {
        throw std::runtime_error("Input data size " +
                                  std::to_string(inputData.size()) +
                                  " is smaller than expected from header " +
                                  std::to_string(expectedSize));
    }

    spdlog::debug("Loading camera size: {}", header.cameraSize);
    spdlog::debug("Loading reference color size: {}", header.referenceColorSize);
    spdlog::debug("Loading residual color size: {}", header.residualColorSize);
    spdlog::debug("Loading geometry size: {}", header.geometrySize);

    // Read camera data
    cameraPose.loadFromMemory(ptr, header.cameraSize);
    copyPoseToCamera(remoteCamera);
    ptr += header.cameraSize;

    // Read reference color data
    referenceColorData.resize(header.referenceColorSize);
    std::memcpy(referenceColorData.data(), ptr, header.referenceColorSize);
    ptr += header.referenceColorSize;

    residualColorData.resize(header.residualColorSize);
    std::memcpy(residualColorData.data(), ptr, header.residualColorSize);
    ptr += header.residualColorSize;

    // Read geometry data
    geometryData.resize(header.geometrySize);
    std::memcpy(geometryData.data(), ptr, header.geometrySize);
    ptr += header.geometrySize;

    // Run async
    FileIO::flipVerticallyOnLoad(true);
    auto refFuture = std::async(std::launch::async, [this]() {
        int width, height, channels;
        unsigned char* data = FileIO::loadImageFromMemory(referenceColorData.data(), referenceColorData.size(), &width, &height, &channels);
        return std::make_tuple(data, width, height, channels);
    });
    std::future<std::tuple<unsigned char*, int, int, int>> resFuture;
    if (header.residualColorSize > 0) {
        resFuture = std::async(std::launch::async, [this]() {
            int width, height, channels;
            unsigned char* data = FileIO::loadImageFromMemory(residualColorData.data(), residualColorData.size(), &width, &height, &channels);
            return std::make_tuple(data, width, height, channels);
        });
    }

    // Load QuadFrame
    if (header.frameType == FrameType::REFERENCE) {
        referenceFrame.loadFromMemory(geometryData);
    }
    else {
        residualFrame.loadFromMemory(geometryData);
    }
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    updateGeometry(header.frameType);

    // Wait for async to finish
    if (header.residualColorSize > 0) {
        auto [resData, resWidth, resHeight, resChannels] = resFuture.get();
        residualColorTexture.resize(resWidth, resHeight);
        residualColorTexture.loadFromData(resData);
        FileIO::freeImage(resData);
    }

    auto [refData, refWidth, refHeight, refChannels] = refFuture.get();
    referenceColorTexture.resize(refWidth, refHeight);
    referenceColorTexture.loadFromData(refData);
    FileIO::freeImage(refData);

    return header.frameType;
}

void QuadsReceiver::updateGeometry(FrameType frameType) {
    const glm::vec2& gBufferSize = quadSet.getSize();
    if (frameType == FrameType::REFERENCE) {
        // Decompress proxies (asynchronous)
        auto offsetsFuture = referenceFrame.decompressDepthOffsets(uncompressedOffsets);
        auto quadsFuture = referenceFrame.decompressQuads(uncompressedQuads);
        quadsFuture.get(); offsetsFuture.get();

        // Transfer updated proxies to GPU for reconstruction
        auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
        referenceFrame.numQuads = sizes.numQuads;
        referenceFrame.numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct mesh using proxies
        double startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
        stats.timeToDecompressMs += referenceFrame.getTimeToDecompress();

        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }
    else {
        // Decompress proxies (asynchronous)
        auto offsetsUpdatedFuture = residualFrame.decompressUpdatedDepthOffsets(uncompressedOffsetsUpdated);
        auto offsetsRevealedFuture = residualFrame.decompressRevealedDepthOffsets(uncompressedOffsetsRevealed);
        auto quadsUpdatedFuture = residualFrame.decompressUpdatedQuads(uncompressedQuadsUpdated);
        auto quadsRevealedFuture = residualFrame.decompressRevealedQuads(uncompressedQuadsRevealed);
        quadsUpdatedFuture.get(); offsetsUpdatedFuture.get();

        // Transfer updated proxies to GPU for reconstruction
        auto sizesUpdated = quadSet.copyFromCPU(uncompressedQuadsUpdated, uncompressedOffsetsUpdated, false);
        residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
        residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
        stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

        // Using GPU buffers, update reference frame mesh using proxies
        double startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize, false /* is not reference frame */);
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
        stats.sizes += sizesUpdated + sizesRevealed;
        stats.timeToDecompressMs += referenceFrame.getTimeToDecompress() + residualFrame.getTimeToDecompress();
    }
}
