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
    FrameRenderTarget referenceFrameRT;
    ReferenceFrame referenceFrame;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodes;
    int currMeshIndex = 0, prevMeshIndex = 1;

    // Residual frame
    FrameRenderTarget residualFrameRT;
    // Render target to hold updated/masked depth and normals for residual frames
    FrameRenderTarget residualFrameMaskRT;
    ResidualFrame residualFrame;
    QuadMesh residualFrameMesh;
    Node residualFrameNodeLocal;

    // Local objects
    std::vector<Node> referenceFrameNodesLocal;
    std::vector<Node> referenceFrameWireframesLocal;
    Node residualFrameWireframesLocal;

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
        double totaltimeToCreateMeshMs = 0.0;
        double totalAppendQuadsTime = 0.0;
        double totalFillQuadsIndiciesTime = 0.0;
        double totalCreateVertIndTime = 0.0;
        double totalGenDepthTime = 0.0;
        double totalCompressTime = 0.0;
        QuadSet::Sizes totalSizes;
    } stats;

    QuadsStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer, Scene& remoteScene,
        const std::string& receiverURL = "");
    ~QuadsStreamer() = default;

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);

    void generateFrame(
        DeferredRenderer& remoteRenderer, Scene& remoteScene, const PerspectiveCamera& remoteCamera,
        bool createResidualFrame, bool showNormals = false, bool showDepth = false);
    void sendProxies(const PerspectiveCamera& remoteCamera, bool createResidualFrame);

    size_t writeToFile(const PerspectiveCamera& remoteCamera, const Path& outputPath);
    size_t writeToMemory(const PerspectiveCamera& remoteCamera, bool isResidualFrame, std::vector<char>& outputData);

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
    FrameGenerator frameGenerator;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;

    PerspectiveCamera remoteCameraPrev;

    Pose cameraPose;

    // Scenes with resulting meshes
    std::vector<Scene> meshScenes;

    // Holds a copy of the current reference and residual frame
    FrameRenderTarget referenceCopyRT;
    FrameRenderTarget residualCopyRT;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;

    std::vector<char> compressedData;

    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;
};

} // namespace quasar

#endif // QUADS_SIMULATOR_H
