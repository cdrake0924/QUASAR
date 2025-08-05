#ifndef FRAME_GENERATOR_H
#define FRAME_GENERATOR_H

#include <Renderers/DeferredRenderer.h>
#include <RenderTargets/FrameRenderTarget.h>

#include <Quads/QuadFrame.h>
#include <Quads/QuadsGenerator.h>
#include <Quads/MeshFromQuads.h>

namespace quasar {

class FrameGenerator {
public:
    QuadsGenerator quadsGenerator;

    FrameGenerator(QuadFrame& quadFrame, DeferredRenderer& remoteRenderer, Scene& remoteScene);

    struct Stats {
        double timeToCreateProxiesMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        double timeToGenerateQuadsMs = 0.0;
        double timeToSimplifyQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
        double timeToAppendQuadsMs = 0.0;
        double timeToFillQuadIndicesMs = 0.0;
        double timeToCreateVertIndMs = 0.0;
        double timeToRenderMaskMs = 0.0;
        double timeToCompress = 0.0f;
    } stats;

    QuadFrame::Sizes generateRefFrame(
        const FrameRenderTarget& frameRT,
        const PerspectiveCamera& remoteCamera,
        const Mesh& mesh);

    QuadFrame::Sizes generateResFrame(
        Scene& currScene, Scene& prevScene,
        FrameRenderTarget& frameRT, FrameRenderTarget& maskFrameRT,
        const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
        const Mesh& currMesh, const Mesh& maskMesh);

    std::pair<MeshFromQuads::BufferSizes, MeshFromQuads::BufferSizes> getBufferSizes() const {
        return { meshFromQuads.getBufferSizes(), meshFromQuadsMask.getBufferSizes() };
    }

private:
    QuadFrame& quadFrame;

    MeshFromQuads meshFromQuads;
    MeshFromQuads meshFromQuadsMask;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
};

} // namespace quasar

#endif // FRAME_GENERATOR_H
