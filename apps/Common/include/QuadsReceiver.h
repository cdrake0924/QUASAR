#ifndef QUADS_RECEIVER_H
#define QUADS_RECEIVER_H

#include <deque>

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>
#include <Networking/DataReceiverTCP.h>

namespace quasar {

class QuadsReceiver : public DataReceiverTCP {
public:
    struct Header {
        uint32_t cameraSize;
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

    std::string streamerURL;
    ReferenceFrame frame;

    QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& streamerURL = "");
    ~QuadsReceiver() = default;

    QuadMesh& getMesh();
    PerspectiveCamera& getRemoteCamera();

    void copyPoseToCamera(PerspectiveCamera& camera);

    void onDataReceived(const std::vector<char>& data) override;
    void processFrames();

    void loadFromFiles(const Path& dataPath);
    void loadFromMemory(const std::vector<char>& inputData);

private:
    void updateGeometry();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    Pose cameraPose;

    QuadMaterial quadMaterial;
    Texture colorTexture;
    QuadMesh mesh;

    std::mutex m;
    std::deque<std::vector<char>> frames;

    std::vector<char> uncompressedQuads, uncompressedOffsets;
    std::vector<unsigned char> colorData;
    std::vector<char> geometryData;
};

} // namespace quasar

#endif // QUADS_RECEIVER_H
