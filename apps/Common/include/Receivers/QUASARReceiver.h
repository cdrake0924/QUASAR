#ifndef QUASAR_RECEIVER_H
#define QUASAR_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <CameraPose.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Networking/DataReceiverTCP.h>
#include <Receivers/VideoTexture.h>

namespace quasar {

class QUASARReceiver : public DataReceiverTCP {
public:
    struct Params {
        uint32_t numLayers;
        float viewSphereDiameter;
        float wideFOV;
    };

    struct Header {
        pose_id_t poseID;
        QuadFrame::FrameType frameType;
        uint32_t cameraSize;
        Params params;
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

    uint maxLayers;
    float viewSphereDiameter;

    VideoTexture atlasVideoTexture;

    QUASARReceiver(QuadSet& quadSet, uint maxLayers, const std::string& videoURL = "", const std::string& proxiesURL = "");
    QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide, const std::string& videoURL = "", const std::string& proxiesURL = "");
    ~QUASARReceiver() = default;

    QuadMesh& getMesh(int layer) { return meshes[layer]; }
    QuadMesh& getResidualMesh() { return residualFrameMesh; }
    PerspectiveCamera& getRemoteCamera() { return remoteCamera; }
    PerspectiveCamera& getRemoteCameraPrev() { return remoteCameraPrev; }
    PerspectiveCamera& getremoteCameraWideFOV() { return remoteCameraWideFOV; }
    void copyPoseToCamera(PerspectiveCamera& camera) {
        camera.setViewMatrix(remoteCamera.getViewMatrix());
        camera.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    }

    void setViewSphereDiameter(float viewSphereDiameter) { this->viewSphereDiameter = viewSphereDiameter; }

    QuadFrame::FrameType loadFromFiles(const Path& dataPath);
    QuadFrame::FrameType loadFromMemory(const std::vector<char>& inputData);

    QuadFrame::FrameType recvData();

private:
    QuadSet& quadSet;
    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraWideFOV;
    PerspectiveCamera remoteCameraPrev;

    std::vector<QuadMesh> meshes;
    QuadMesh residualFrameMesh;

    struct BufferPool {
        std::vector<std::vector<char>> uncompressedQuads, uncompressedOffsets;
        std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;

        BufferPool(const glm::vec2& gBufferSize, int maxLayers, size_t maxProxiesPerMesh = MAX_PROXIES_PER_MESH) {
            uncompressedQuads.resize(maxLayers);
            uncompressedOffsets.resize(maxLayers);

            const size_t quadsBytes   = sizeof(uint) + maxProxiesPerMesh * sizeof(QuadMapDataPacked);
            const size_t offsetsBytes = static_cast<size_t>(gBufferSize.x * gBufferSize.y) * 4 * sizeof(uint16_t);

            for (int layer = 0; layer < maxLayers; ++layer) {
                size_t adjustedQuadsBytes = (layer == 0) ? quadsBytes :
                                            (layer == maxLayers - 1) ? quadsBytes / 2 : quadsBytes / (layer * 4);
                uncompressedQuads[layer].resize(adjustedQuadsBytes);
                uncompressedOffsets[layer].resize(offsetsBytes);
            }

            uncompressedQuadsRevealed.resize(quadsBytes / 4);
            uncompressedOffsetsRevealed.resize(offsetsBytes);
        }
    };

    struct Frame {
        pose_id_t poseID = -1;
        QuadFrame::FrameType frameType;
        Pose cameraPose;
        BufferPool& buffers;

        Frame(BufferPool& buffers) : frameType(QuadFrame::FrameType::NONE), buffers(buffers) {}
        ~Frame() = default;

        size_t decompressReferenceFrames(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                         std::vector<ReferenceFrame>& referenceFrames) {
            std::vector<std::future<size_t>> futures;
            for (int layer = 0; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(buffers.uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(buffers.uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }

        size_t decompressReferenceAndResidualFrames(std::unique_ptr<BS::thread_pool<>>& threadPool,
                                                    std::vector<ReferenceFrame>& referenceFrames,
                                                    ResidualFrame& residualFrame) {
            std::vector<std::future<size_t>> futures;
            futures.reserve(4 + 2 * (referenceFrames.size() - 1));

            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedDepthOffsets(buffers.uncompressedOffsets[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedDepthOffsets(buffers.uncompressedOffsetsRevealed);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressUpdatedQuads(buffers.uncompressedQuads[0]);
            }));
            futures.emplace_back(threadPool->submit_task([&]() {
                return residualFrame.decompressRevealedQuads(buffers.uncompressedQuadsRevealed);
            }));

            for (int layer = 1; layer < referenceFrames.size(); layer++) {
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressDepthOffsets(buffers.uncompressedOffsets[layer]);
                }));
                futures.emplace_back(threadPool->submit_task([&, layer]() {
                    return referenceFrames[layer].decompressQuads(buffers.uncompressedQuads[layer]);
                }));
            }

            size_t outputSize = 0;
            for (auto& f : futures) outputSize += f.get();
            return outputSize;
        }
    };

    BufferPool bufferPool;

    std::mutex m;
    std::condition_variable cv;
    std::shared_ptr<Frame> frameInUse;
    std::shared_ptr<Frame> framePending;
    std::shared_ptr<Frame> frameFree;

    bool waitUntilReferenceFrame = false;

    std::vector<ReferenceFrame> referenceFrames;
    ResidualFrame residualFrame;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    std::vector<char> geometryData;

    inline const PerspectiveCamera& getCameraToUse(int layer) const {
        return (layer == maxLayers - 1) ? remoteCameraWideFOV : remoteCamera;
    }

    void onDataReceived(const std::vector<char>& data) override;
    QuadFrame::FrameType reconstructFrame(std::shared_ptr<Frame> frame);
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
