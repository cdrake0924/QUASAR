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
    , mesh(quadSet, colorTexture)
    , uncompressedQuads(sizeof(uint) + quadSet.quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
    , uncompressedOffsets(quadSet.depthOffsets.getSize().x * quadSet.depthOffsets.getSize().y * 4 * sizeof(uint16_t))
    , DataReceiverTCP(streamerURL)
{
    remoteCamera.setFovyDegrees(remoteFOV);

    if (!streamerURL.empty()) {
        spdlog::info("Created QuadsReceiver that recvs from URL: {}", streamerURL);
    }
}

QuadMesh& QuadsReceiver::getMesh() {
    return mesh;
}

PerspectiveCamera& QuadsReceiver::getRemoteCamera() {
    return remoteCamera;
}

void QuadsReceiver::copyPoseToCamera(PerspectiveCamera& camera) {
    camera.setProjectionMatrix(cameraPose.mono.proj);
    camera.setViewMatrix(cameraPose.mono.view);
}

void QuadsReceiver::onDataReceived(const std::vector<char>& data) {
    std::lock_guard<std::mutex> lock(m);
    frames.push_back(std::move(data));
}

void QuadsReceiver::processFrames() {
    std::lock_guard<std::mutex> lock(m);
    while (!frames.empty()) {
        auto data = frames.front();
        frames.pop_front();
        loadFromMemory(data);
    }
}

void QuadsReceiver::loadFromFiles(const Path& dataPath) {
    stats = { 0 };
    double startTime = timeutils::getTimeMicros();

    std::string cameraFileName = dataPath / "camera.bin";
    cameraPose.loadFromFile(cameraFileName);
    copyPoseToCamera(remoteCamera);

    std::string colorFileName = dataPath / "color.jpg";
    colorTexture.loadFromFile(colorFileName, true, false);

    frame.loadFromFiles(dataPath);
    stats.loadTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    updateGeometry();
}

void QuadsReceiver::loadFromMemory(const std::vector<char>& inputData) {
    stats = { 0 };
    double startTime = timeutils::getTimeMicros();

    const char* ptr = inputData.data();
    Header header;
    std::memcpy(&header, ptr, sizeof(Header));
    ptr += sizeof(Header);

    if (inputData.size() < header.cameraSize + header.colorSize + header.geometrySize) {
        throw std::runtime_error("Input data size " +
                                  std::to_string(inputData.size()) +
                                  " is smaller than expected from header " +
                                  std::to_string(header.colorSize + header.geometrySize));
    }

    cameraPose.loadFromMemory(ptr, header.cameraSize);
    copyPoseToCamera(remoteCamera);
    ptr += header.cameraSize;

    colorData.resize(header.colorSize);
    std::memcpy(colorData.data(), ptr, header.colorSize);
    ptr += header.colorSize;

    geometryData.resize(header.geometrySize);
    std::memcpy(geometryData.data(), ptr, header.geometrySize);
    ptr += header.geometrySize;

    // Run async
    auto future = std::async(std::launch::async, [this]() {
        FileIO::flipVerticallyOnLoad(true);
        int width, height, channels;
        unsigned char* data = FileIO::loadImageFromMemory(colorData.data(), colorData.size(), &width, &height, &channels);
        return std::make_tuple(data, width, height, channels);
    });

    frame.loadFromMemory(geometryData);
    stats.loadTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    updateGeometry();

    // Wait for async to finish
    auto [data, width, height, channels] = future.get();
    colorTexture.resize(width, height);
    colorTexture.loadFromData(data);
    FileIO::freeImage(data);
}

void QuadsReceiver::updateGeometry() {
    double startTime = timeutils::getTimeMicros();
    auto offsetsFuture = frame.decompressDepthOffsets(uncompressedOffsets);
    auto quadsFuture = frame.decompressQuads(uncompressedQuads);
    quadsFuture.get(); offsetsFuture.get();
    stats.decompressTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
    stats.transferTime += quadSet.stats.timeToTransferMs;

    const glm::vec2& gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);
    startTime = timeutils::getTimeMicros();
    mesh.appendQuads(quadSet, gBufferSize);
    mesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
    stats.createMeshTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    auto meshBufferSizes = mesh.getBufferSizes();
    stats.totalTriangles += meshBufferSizes.numIndices / 3;
    stats.sizes += sizes;
}

} // namespace quasar
