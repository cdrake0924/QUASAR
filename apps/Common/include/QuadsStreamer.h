#ifndef QUADS_SIMULATOR_H
#define QUADS_SIMULATOR_H

#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <QuadsReceiver.h>
#include <CameraPose.h>
#include <Networking/DataStreamerTCP.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QuadsStreamer : public DataStreamerTCP {
public:
    // Reference frame
    FrameRenderTarget refFrameRT;
    ReferenceFrame referenceFrame;
    std::vector<QuadMesh> refFrameMeshes;
    std::vector<Node> refFrameNodes;
    int currMeshIndex = 0, prevMeshIndex = 1;

    // Residual frame
    FrameRenderTarget resFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget resFrameMaskRT;
    ResidualFrame residualFrame;
    QuadMesh resFrameMesh;
    Node resFrameNode;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    std::vector<Node> refFrameWireframesLocal;
    Node resFrameWireframeNodesLocal;

    // Depth point cloud for debugging
    DepthMesh depthMesh;
    Node depthNode;

    std::string receiverURL;

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

    QuadsStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        FrameGenerator& frameGenerator,
        const std::string& receiverURL = "");
    ~QuadsStreamer() = default;

    uint getNumTriangles() const;

    void addMeshesToScene(Scene& localScene);

    void generateFrame(
        DeferredRenderer& remoteRenderer, Scene& remoteScene,
        bool createResidualFrame = false, bool showNormals = false, bool showDepth = false);

    size_t writeToFile(const Path& outputPath);
    size_t writeToMemory(bool isResidualFrame, std::vector<char>& outputData);

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

    const PerspectiveCamera& remoteCamera;
    PerspectiveCamera remoteCameraPrev;

    Pose cameraPose;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;

    // Holds a copy of the current frame
    FrameRenderTarget copyRT;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    std::vector<char> compressedData;

    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;
};

} // namespace quasar

#endif // QUADS_SIMULATOR_H
