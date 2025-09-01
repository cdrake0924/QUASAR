#ifndef QUADS_RECEIVER_H
#define QUADS_RECEIVER_H

#include <array>

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Networking/DataReceiverTCP.h>
#include <Receivers/VideoTexture.h>

namespace quasar {

class QuadsReceiver : public DataReceiverTCP {
public:
    struct Header {
        pose_id_t poseID;
        FrameType frameType;
        uint32_t cameraSize;
        uint32_t geometrySize;
    };

    struct Stats {
        uint totalTriangles = 0;
        double timeToLoadMs = 0.0;
        double timeToDecompressMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    std::string proxiesURL;
    std::string videoURL;

    VideoTexture atlasVideoTexture;

    QuadsReceiver(QuadSet& quadSet, const std::string& videoURL = "", const std::string& proxiesURL = "");
    QuadsReceiver(QuadSet& quadSet, float remoteFOV, const std::string& videoURL = "", const std::string& proxiesURL = "");
    ~QuadsReceiver() = default;

    QuadMesh& getReferenceMesh() { return referenceFrameMesh; }
    QuadMesh& getResidualMesh() { return residualFrameMesh; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    PerspectiveCamera& getRemoteCameraPrev() { return remoteCameraPrev; }
    void copyPoseToCamera(PerspectiveCamera& camera) { frameInUse->cameraPose.copyPoseToCamera(camera); }

    FrameType loadFromFiles(const Path& dataPath);
    FrameType loadFromMemory(const std::vector<char>& data);

    FrameType recvData();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraPrev;

    QuadMesh referenceFrameMesh;
    QuadMesh residualFrameMesh;

    struct Frame {
        pose_id_t poseID;
        FrameType frameType;

        Pose cameraPose;

        std::vector<char> uncompressedQuads, uncompressedOffsets;
        std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;

        Frame(const glm::vec2& gBufferSize) : frameType(FrameType::NONE)
        {
            const size_t quadsBytes = sizeof(uint) + MAX_QUADS_PER_MESH * sizeof(QuadMapDataPacked);
            const size_t offsetsBytes = static_cast<size_t>(gBufferSize.x) * static_cast<size_t>(gBufferSize.y) * 4u * sizeof(uint16_t);

            uncompressedQuads.resize(quadsBytes);
            uncompressedOffsets.resize(offsetsBytes);

            uncompressedQuadsRevealed.resize(quadsBytes / 4);
            uncompressedOffsetsRevealed.resize(offsetsBytes);
        }
        ~Frame() = default;

        void decompressReferenceFrame(std::unique_ptr<BS::thread_pool<>>& threadPool, ReferenceFrame& referenceFrame) {
            // Decompress proxies (asynchronous)
            auto offsetsFuture = threadPool->submit_task([&]() {
                return referenceFrame.decompressDepthOffsets(uncompressedOffsets);
            });
            auto quadsFuture = threadPool->submit_task([&]() {
                return referenceFrame.decompressQuads(uncompressedQuads);
            });
            quadsFuture.get(); offsetsFuture.get();
        }

        void decompressResidualFrame(std::unique_ptr<BS::thread_pool<>>& threadPool, ResidualFrame& residualFrame) {
            // Decompress proxies (asynchronous)
            auto offsetsUpdatedFuture = threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedDepthOffsets(uncompressedOffsets);
            });
            auto offsetsRevealedFuture = threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedDepthOffsets(uncompressedOffsetsRevealed);
            });
            auto quadsUpdatedFuture = threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedQuads(uncompressedQuads);
            });
            auto quadsRevealedFuture = threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedQuads(uncompressedQuadsRevealed);
            });
            quadsUpdatedFuture.get(); offsetsUpdatedFuture.get();
            quadsRevealedFuture.get(); offsetsRevealedFuture.get();
        }
    };

    std::mutex m;
    std::condition_variable cv;
    std::shared_ptr<Frame> frameInUse;
    std::shared_ptr<Frame> framePending;
    std::shared_ptr<Frame> frameFree;

    ReferenceFrame referenceFrame;
    ResidualFrame residualFrame;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    std::vector<char> geometryData;

    void onDataReceived(const std::vector<char>& data) override;
    FrameType loadFromFrame(std::shared_ptr<Frame> frame);
};

} // namespace quasar

#endif // QUADS_RECEIVER_H
