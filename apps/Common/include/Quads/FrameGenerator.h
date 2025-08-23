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

    QuadsGenerator quadsGenerator;

    FrameGenerator(QuadSet& quadSet);

    void createReferenceFrame(
        const FrameRenderTarget& referenceFrameRT,
        const PerspectiveCamera& remoteCamera,
        QuadMesh& mesh,
        ReferenceFrame& resultFrame,
        bool compress = true);

    void updateResidualRenderTargets(
        FrameRenderTarget& resFrameMaskRT, FrameRenderTarget& resFrameRT,
        DeferredRenderer& remoteRenderer, Scene& remoteScene,
        Scene& currMeshScene, Scene& prevMeshScene, // Scenes that contain the resulting reconstructed meshes
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera);

    void createResidualFrame(
        const FrameRenderTarget& resFrameMaskRT, const FrameRenderTarget& resFrameRT,
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
        QuadMesh& mesh, QuadMesh& maskMesh,
        ResidualFrame& resultFrame,
        bool compress = true);

private:
    QuadSet& quadSet;

    ZSTDCodec refQuadsCodec;
    ZSTDCodec refOffsetsCodec;

    ZSTDCodec resQuadsUpdatedCodec;
    ZSTDCodec resOffsetsUpdatedCodec;
    ZSTDCodec resQuadsRevealedCodec;
    ZSTDCodec resOffsetsRevealedCodec;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;
    std::vector<char> uncompressedQuadsRevealed, uncompressedOffsetsRevealed;
    std::vector<char> uncompressedQuadsUpdated, uncompressedOffsetsUpdated;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
