#include <cstring>
#include <stdexcept>

#include <Utils/FileIO.h>
#include <Utils/TimeUtils.h>
#include <Receivers/QuadsReceiver.h>

using namespace quasar;

QuadsReceiver::QuadsReceiver(QuadSet& quadSet, const std::string& streamerURL)
    : quadSet(quadSet)
    , streamerURL(streamerURL)
    , remoteCamera(quadSet.getSize())
    , atlasTexture({
        .internalFormat = GL_RGB,
        .format = GL_RGB,
        .type = GL_UNSIGNED_BYTE,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , referenceFrameMesh(quadSet, atlasTexture, glm::vec4(0.0f, 0.0f, 0.5f, 1.0f))
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, atlasTexture, glm::vec4(0.5f, 0.0f, 1.0f, 1.0f), MAX_QUADS_PER_MESH / 4)
    , DataReceiverTCP(streamerURL)
{
    remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    frameInUse = std::make_shared<Frame>(quadSet.getSize());
    framePending = std::make_shared<Frame>(quadSet.getSize());

    threadPool = std::make_unique<BS::thread_pool<>>(5); // 1 thread for color, 4 threads for proxies

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
    loadFromMemory(data);
}

FrameType QuadsReceiver::recvData() {
    if (streamerURL.empty()) {
        return FrameType::NONE;
    }

    // Wait for a written to frame
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        if (!framePending) {
            return FrameType::NONE;
        }
        frame = framePending;
        framePending.reset();
        frameInUse = frame;
    }

    // Reset frame
    FrameType type = loadFromFrame(frame);
    {
        std::lock_guard<std::mutex> lock(m);
        frameFree = frame;
    }
    cv.notify_one();

    return type;
}

FrameType QuadsReceiver::loadFromMemory(const std::vector<char>& inputData) {
    double startTime = timeutils::getTimeMicros();

    // Unpack frame
    const char* ptr = inputData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    // Sanity check
    size_t expectedSize = sizeof(Header) +
                          header.cameraSize +
                          header.colorSize +
                          header.geometrySize;
    if (inputData.size() < expectedSize) {
        throw std::runtime_error("Input data size " +
                                  std::to_string(inputData.size()) +
                                  " is smaller than expected from header " +
                                  std::to_string(expectedSize));
    }

    spdlog::debug("Loading camera size: {}", header.cameraSize);
    spdlog::debug("Loading color size: {}", header.colorSize);
    spdlog::debug("Loading geometry size: {}", header.geometrySize);

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

    // Wait for a free frame
    std::shared_ptr<Frame> frame;
    {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]() { return frameFree != nullptr; });
        frame = frameFree;
        frameFree.reset();
    }
    frame->frameType = header.frameType;

    stats.timeToLoadMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    startTime = timeutils::getTimeMicros();

    // Run JPG decompression (asynchronous)
    auto colFuture = threadPool->submit_task([&]() mutable {
        FileIO::flipVerticallyOnLoad(true);
        int width, height, channels;
        unsigned char* data = FileIO::loadImageFromMemory(colorData.data(), colorData.size(), &width, &height, &channels);
        return std::make_tuple(data, width, height, channels);
    });

    if (header.frameType == FrameType::REFERENCE) {
        referenceFrame.loadFromMemory(geometryData);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        frame->decompressReferenceFrame(threadPool, referenceFrame);
    }
    else {
        residualFrame.loadFromMemory(geometryData);
        stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        frame->decompressResidualFrame(threadPool, residualFrame);
    }

    auto [colData, colWidth, colHeight, colChannels] = colFuture.get();
    frame->width = colWidth;
    frame->height = colHeight;
    frame->colorData = colData;

    stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Signal that frame is ready
    {
        std::lock_guard<std::mutex> lock(m);
        framePending = frame;
    }
    cv.notify_one();

    return frame->frameType;
}

FrameType QuadsReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };

    double startTime = timeutils::getTimeMicros();

    // Read camera data
    Path cameraFileNamePrev = dataPath / "camera_prev.bin";
    cameraPose.loadFromFile(cameraFileNamePrev);
    copyPoseToCamera(remoteCamera);

    // Read color data
    Path colorFileName = dataPath / "color.jpg";
    atlasTexture.loadFromFile(colorFileName, true, false);

    // Load reference frame
    referenceFrame.loadFromFiles(dataPath);
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    frameInUse->frameType = FrameType::REFERENCE;
    frameInUse->decompressReferenceFrame(threadPool, referenceFrame);

    // Update GPU buffers
    updateGeometry(frameInUse);

    startTime = timeutils::getTimeMicros();

    // Read previous camera data
    Path cameraFileName = dataPath / "camera.bin";
    cameraPose.loadFromFile(cameraFileName);
    copyPoseToCamera(remoteCamera);

    // Load residual frame
    residualFrame.loadFromFiles(dataPath);
    stats.timeToLoadMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    frameInUse->frameType = FrameType::RESIDUAL;
    frameInUse->decompressResidualFrame(threadPool, residualFrame);

    // Update GPU buffers
    updateGeometry(frameInUse);

    return frameInUse->frameType;
}

FrameType QuadsReceiver::loadFromFrame(std::shared_ptr<Frame> frame) {
    atlasTexture.resize(frame->width, frame->height);
    atlasTexture.loadFromData(frame->colorData);

    updateGeometry(frame);
    spdlog::debug("Reconstructing {} Frame...", frame->frameType == FrameType::REFERENCE ? "Reference" : "Residual");
    return frame->frameType;
}

void QuadsReceiver::updateGeometry(std::shared_ptr<Frame> frame) {
    const glm::vec2& gBufferSize = quadSet.getSize();
    if (frame->frameType == FrameType::REFERENCE) {
        // Transfer updated proxies to GPU for reconstruction
        auto sizes = quadSet.loadFromMemory(frame->uncompressedQuads, frame->uncompressedOffsets);
        referenceFrame.numQuads = sizes.numQuads;
        referenceFrame.numDepthOffsets = sizes.numDepthOffsets;
        stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

        // Using GPU buffers, reconstruct mesh using proxies
        double startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto refMeshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles = refMeshBufferSizes.numIndices / 3;
        stats.sizes = sizes;

        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }
    else {
        // Transfer updated proxies to GPU for reconstruction
        auto sizesUpdated = quadSet.loadFromMemory(frame->uncompressedQuads, frame->uncompressedOffsets);
        residualFrame.numQuadsUpdated = sizesUpdated.numQuads;
        residualFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
        stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

        // Using GPU buffers, update reference frame mesh using proxies
        double startTime = timeutils::getTimeMicros();
        referenceFrameMesh.appendQuads(quadSet, gBufferSize, false /* is not reference frame */);
        referenceFrameMesh.createMeshFromProxies(quadSet, gBufferSize, remoteCameraPrev);
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // This will also wait for the GPU to finish
        auto refMeshBufferSizes = referenceFrameMesh.getBufferSizes();
        stats.totalTriangles = refMeshBufferSizes.numIndices / 3;

        // Transfer revealed proxies to GPU for reconstruction
        auto sizesRevealed = quadSet.loadFromMemory(frame->uncompressedQuadsRevealed, frame->uncompressedOffsetsRevealed);
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
    }
}
