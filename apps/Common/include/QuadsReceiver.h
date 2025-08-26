#ifndef QUADS_RECEIVER_H
#define QUADS_RECEIVER_H

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

class QuadsReceiver {
public:
    struct Header {
        uint16_t cameraSize;
        uint32_t colorSize;
        uint32_t geometrySize;
    };

    struct Stats {
        uint totalTriangles = 0;
        double loadTime = 0.0;
        double decompressTime = 0.0;
        double transferTime = 0.0;
        double createMeshTime = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    ReferenceFrame frame;

    QuadsReceiver(QuadSet& quadSet, float remoteFOV)
        : quadSet(quadSet)
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
    {
        remoteCamera.setFovyDegrees(remoteFOV);
    }
    ~QuadsReceiver() = default;

    QuadMesh& getMesh() {
        return mesh;
    }

    PerspectiveCamera& getRemoteCamera() {
        return remoteCamera;
    }

    void copyPoseToCamera(PerspectiveCamera& camera) {
        camera.setProjectionMatrix(cameraPose.mono.proj);
        camera.setViewMatrix(cameraPose.mono.view);
    }

    void loadFromFiles(const Path& dataPath) {
        stats = { 0 };

        double startTime = timeutils::getTimeMicros();

        // Load camera data
        std::string cameraFileName = dataPath / "camera.bin";
        cameraPose.loadFromFile(cameraFileName);
        copyPoseToCamera(remoteCamera);

        // Load texture
        std::string colorFileName = dataPath / "color.jpg";
        colorTexture.loadFromFile(colorFileName, true, false);

        // Load quads and depth offsets from files
        frame.loadFromFiles(dataPath);
        stats.loadTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Decompress and update mesh
        updateGeometry();
    }
    // void loadFromFiles(const Path& dataPath) {
    //     stats = { 0 };
    //     std::vector<char> data = FileIO::loadBinaryFile(dataPath / "compressed.bin.zstd");
    //     loadFromMemory(data);
    // }

    void loadFromMemory(const std::vector<char>& inputData) {
        stats = { 0 };

        double startTime = timeutils::getTimeMicros();

        const char* ptr = inputData.data();

        // Read header
        Header header;
        std::memcpy(&header, ptr, sizeof(Header));
        ptr += sizeof(Header);

        // Sanity check
        if (inputData.size() < header.cameraSize + header.colorSize + header.geometrySize) {
            throw std::runtime_error("Input data size " +
                                      std::to_string(inputData.size()) +
                                      " is smaller than expected from header " +
                                      std::to_string(header.colorSize + header.geometrySize));
        }

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

        updateColorTexture();
        frame.loadFromMemory(geometryData);
        stats.loadTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Decompress and update mesh
        updateGeometry();
    }

    void updateColorTexture() {
        int colorWidth, colorHeight, colorChannels;
        FileIO::flipVerticallyOnLoad(true);
        unsigned char* data = FileIO::loadImageFromMemory(colorData.data(), colorData.size(), &colorWidth, &colorHeight, &colorChannels);
        colorTexture.resize(colorWidth, colorHeight);
        colorTexture.loadFromData(data);
        FileIO::freeImage(data);
    }

    void updateGeometry() {
        double startTime = timeutils::getTimeMicros();
        auto offsetsFuture = frame.decompressDepthOffsets(uncompressedOffsets);
        auto quadsFuture = frame.decompressQuads(uncompressedQuads);
        quadsFuture.get(); offsetsFuture.get();
        stats.decompressTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        // Copy data to GPU
        auto sizes = quadSet.copyFromCPU(uncompressedQuads, uncompressedOffsets);
        stats.transferTime += quadSet.stats.timeToTransferMs;

        // Update mesh
        const glm::vec2& gBufferSize = glm::vec2(colorTexture.width, colorTexture.height);
        startTime = timeutils::getTimeMicros();
        mesh.appendQuads(quadSet, gBufferSize);
        mesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
        stats.createMeshTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto meshBufferSizes = mesh.getBufferSizes();
        stats.totalTriangles += meshBufferSizes.numIndices / 3;
        stats.sizes += sizes;
    }

private:
    QuadSet& quadSet;

    PerspectiveCamera remoteCamera;
    Pose cameraPose;

    QuadMaterial quadMaterial;
    Texture colorTexture;
    QuadMesh mesh;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;

    std::vector<unsigned char> colorData;
    std::vector<char> geometryData;
};

} // namespace quasar

#endif // QUADS_RECEIVER_H
