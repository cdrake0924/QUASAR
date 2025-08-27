#ifndef QUADS_RECEIVER_H
#define QUADS_RECEIVER_H

#include <deque>

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Networking/DataReceiverTCP.h>

namespace quasar {

class QuadsReceiver : public DataReceiverTCP {
public:
    struct Header {
        FrameType frameType;
        uint32_t cameraSize;
        uint32_t referenceColorSize;
        uint32_t residualColorSize;
        uint32_t geometrySize;
    };

    struct Stats {
        uint totalTriangles = 0;
        double loadTime = 0.0;
        double timeToDecompressMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    std::string streamerURL;

    ReferenceFrame referenceFrame;
    ResidualFrame residualFrame;

    QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& streamerURL = "");
    ~QuadsReceiver() = default;

    QuadMesh& getReferenceMesh();
    QuadMesh& getResidualMesh();

    void copyPoseToCamera(PerspectiveCamera& camera);

    void onDataReceived(const std::vector<char>& data) override;

    FrameType loadFromFiles(const Path& dataPath);
    FrameType loadFromMemory(const std::vector<char>& inputData);

    FrameType recvData();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraPrev;
    Pose cameraPose;

    Texture referenceColorTexture;
    QuadMesh referenceFrameMesh;
    Texture residualColorTexture;
    QuadMesh residualFrameMesh;

    std::mutex m;
    std::deque<std::vector<char>> frames;

    std::vector<unsigned char> referenceColorData;
    std::vector<unsigned char> residualColorData;
    std::vector<char> geometryData;
    std::vector<char> uncompressedQuads, uncompressedOffsets;
    std::vector<char> uncompressedQuadsUpdated, uncompressedOffsetsUpdated;
    std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;

    void updateGeometry(FrameType frameType);
};

} // namespace quasar

#endif // QUADS_RECEIVER_H
