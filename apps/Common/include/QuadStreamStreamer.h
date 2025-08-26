#ifndef QUADSTREAM_SIMULATOR_H
#define QUADSTREAM_SIMULATOR_H

#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QuadStreamStreamer {
public:
    uint maxViews;

    uint maxVertices = MAX_QUADS_PER_MESH * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    uint maxIndices = MAX_QUADS_PER_MESH * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    uint maxVerticesDepth;

    // Reference frame -- QS has no temporal compression (i.e. no residual frames)
    std::vector<FrameRenderTarget> refFrameRTs;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> refFrameMeshes;
    std::vector<Node> refFrameNodesRemote;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    std::vector<Node> refFrameWireframesLocal;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshes;
    std::vector<Node> depthNodes;

    struct Stats {
        double totalRenderTime = 0.0;
        double totalCreateProxiesTime = 0.0;
        double totalGenQuadMapTime = 0.0;
        double totalSimplifyTime = 0.0;
        double totalGatherQuadsTime = 0.0;
        double totalCreateMeshTime = 0.0;
        double totalAppendQuadsTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;
        QuadSet::Sizes totalSizes;
    } stats;

    QuadStreamStreamer(
        QuadSet& quadSet,
        uint maxViews,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        FrameGenerator& frameGenerator);
    ~QuadStreamStreamer() = default;

    uint getNumTriangles() const;

    void addMeshesToScene(Scene& localScene);

    void generateFrame(
        const std::vector<PerspectiveCamera> remoteCameras, Scene& remoteScene,
        DeferredRenderer& remoteRenderer,
        bool showNormals = false, bool showDepth = false);

    size_t writeToFile(const Path& outputPath);

private:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary view color is yellow
        glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),
        glm::vec4(1.0f, 0.5f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 1.0f, 1.0f, 1.0f),
        glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.5f, 0.0f, 1.0f),
        glm::vec4(0.0f, 0.0f, 0.5f, 1.0f),
        glm::vec4(0.5f, 0.0f, 0.5f, 1.0f),
    };

    QuadSet& quadSet;
    FrameGenerator& frameGenerator;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;

    // Scenes with resulting meshes
    Scene meshScene;

    // Holds a copies of the current frame
    std::vector<FrameRenderTarget> copyRTs;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;
};

} // namespace quasar

#endif // QUADSTREAM_SIMULATOR_H
