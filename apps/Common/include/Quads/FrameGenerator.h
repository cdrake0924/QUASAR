#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadsGenerator.h>
#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Codec/ZSTDCodec.h>

namespace quasar {

class FrameGenerator {
public:
    struct Stats {
        double timeToCreateQuadsMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        double timeToGenerateQuadsMs = 0.0;
        double timeToSimplifyQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
        double timeToAppendQuadsMs = 0.0;
        double timeToFillQuadIndicesMs = 0.0;
        double timeToCreateVertIndMs = 0.0;
        double timeToUpdateRTsMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCompressMs = 0.0;
    } stats;

    FrameGenerator(QuadSet& quadSet);

    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return quadsGenerator; }

    void createReferenceFrame(
        const FrameRenderTarget& referenceFrameRT,
        const PerspectiveCamera& remoteCamera,
        QuadMesh& mesh,
        ReferenceFrame& referenceFrame,
        bool compress = true);

    void updateResidualRenderTargets(
        FrameRenderTarget& residualFrameMaskRT, FrameRenderTarget& residualFrameRT,
        DeferredRenderer& remoteRenderer, Scene& remoteScene,
        Scene& currMeshScene, Scene& prevMeshScene, // Scenes that contain the resulting reconstructed meshes
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev);

    void createResidualFrame(
        const FrameRenderTarget& residualFrameMaskRT, const FrameRenderTarget& residualFrameRT,
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& remoteCameraPrev,
        QuadMesh& referenceMesh, QuadMesh& residualMesh,
        ResidualFrame& residualFrame,
        bool compress = true);

private:
    QuadSet& quadSet;
    std::shared_ptr<QuadsGenerator> quadsGenerator;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
    std::vector<char> uncompressedQuadsUpdated, uncompressedOffsetsUpdated;
    std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
