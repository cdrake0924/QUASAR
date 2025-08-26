#ifndef QUASAR_SIMULATOR_H
#define QUASAR_SIMULATOR_H

#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Renderers/DepthPeelingRenderer.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QUASARStreamer {
public:
    uint maxLayers;

    uint maxVertices = MAX_QUADS_PER_MESH * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    uint maxIndices = MAX_QUADS_PER_MESH * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    uint maxVerticesDepth;

    // Reference frame
    FrameRenderTarget refFrameRT;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;
    std::vector<Node> refFrameWireframesLocal;
    int currMeshIndex = 0, prevMeshIndex = 1;

    // Residual frame -- we only create the residuals to the visible layer
    FrameRenderTarget resFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget resFrameMaskRT;
    std::vector<ResidualFrame> residualFrames;
    QuadMesh resFrameMesh;
    Node resFrameNode;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    Node resFrameWireframeNodesLocal;

    // Hidden layers
    std::vector<FrameRenderTarget> frameRTsHidLayer;
    std::vector<QuadMesh> meshesHidLayer;
    std::vector<Node> nodesHidLayer;
    std::vector<Node> wireframesHidLayer;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshsHidLayer;
    std::vector<Node> depthNodesHidLayer;

    // Wide fov
    std::vector<Node> wideFovNodes;

    DepthMesh depthMesh;
    Node depthNode;

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

    QUASARStreamer(
        QuadSet& quadSet,
        uint maxLayers,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        FrameGenerator& frameGenerator);
    ~QUASARStreamer() = default;

    uint getNumTriangles() const;

    void addMeshesToScene(Scene& localScene);

    void generateFrame(
        const PerspectiveCamera& remoteCameraCenter, const PerspectiveCamera& remoteCameraWideFov, Scene& remoteScene,
        DeferredRenderer& remoteRenderer,
        DepthPeelingRenderer& remoteRendererDP,
        bool createResidualFrame = false, bool showNormals = false, bool showDepth = false);

    size_t writeToFile(const Path& outputPath);

private:
    const std::vector<glm::vec4> colors = {
        glm::vec4(1.0f, 1.0f, 0.0f, 1.0f), // primary layer color is yellow
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

    PerspectiveCamera remoteCameraPrev;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;
    Scene sceneWideFov;

    // Holds a copies of the current frame
    std::vector<FrameRenderTarget> copyRTs;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;
};

} // namespace quasar

#endif // QUASAR_SIMULATOR_H
