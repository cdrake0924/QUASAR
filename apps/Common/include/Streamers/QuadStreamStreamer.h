#ifndef QUADSTREAM_SIMULATOR_H
#define QUADSTREAM_SIMULATOR_H

#include <CameraPose.h>
#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>
#include <Networking/DataStreamerTCP.h>
#include <Streamers/VideoStreamer.h>
#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

namespace quasar {

class QuadStreamStreamer {
public:
    uint maxViews;
    float viewBoxSize;

    // Reference frame -- QS has no temporal compression (i.e. no residual frames)
    std::vector<FrameRenderTarget> referenceFrameRTs;
    std::vector<ReferenceFrame> referenceFrames;
    std::vector<QuadMesh> referenceFrameMeshes;
    std::vector<Node> referenceFrameNodesRemote;

    // Local objects
    std::vector<Node> referenceFrameNodesLocal;
    std::vector<Node> referenceFrameWireframesLocal;

    // Depth point cloud for debugging
    std::vector<DepthMesh> depthMeshes;
    std::vector<Node> depthNodes;

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

    QuadStreamStreamer(
        QuadSet& quadSet,
        uint maxViews,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        PerspectiveCamera& remoteCamera,
        float viewBoxSize,
        float wideFOV);
    ~QuadStreamStreamer() = default;

    uint getNumTriangles() const;
    std::shared_ptr<QuadsGenerator> getQuadsGenerator() { return frameGenerator.getQuadsGenerator(); }

    void addMeshesToScene(Scene& localScene);
    void setViewBoxSize(float viewBoxSize);

    void generateFrame(bool showNormals = false, bool showDepth = false);

    size_t writeToFiles(const Path& outputPath);

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

    const std::vector<glm::vec3> offsets = {
        glm::vec3(-1.0f, +1.0f, -1.0f), // Top-left
        glm::vec3(+1.0f, +1.0f, -1.0f), // Top-right
        glm::vec3(+1.0f, -1.0f, -1.0f), // Bottom-right
        glm::vec3(-1.0f, -1.0f, -1.0f), // Bottom-left
        glm::vec3(-1.0f, +1.0f, +1.0f), // Top-left
        glm::vec3(+1.0f, +1.0f, +1.0f), // Top-right
        glm::vec3(+1.0f, -1.0f, +1.0f), // Bottom-right
        glm::vec3(-1.0f, -1.0f, +1.0f), // Bottom-left
    };

    QuadSet& quadSet;
    FrameGenerator frameGenerator;

    DeferredRenderer& remoteRenderer;
    Scene& remoteScene;
    PerspectiveCamera& remoteCamera;
    std::vector<PerspectiveCamera> remoteCameras;

    // Scenes with resulting meshes
    Scene meshScene;

    // Holds a copies of the current frame
    std::vector<FrameRenderTarget> referenceFrameRTs_noTone;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    QuadMaterial wireframeMaterial;
    QuadMaterial maskWireframeMaterial;
};

} // namespace quasar

#endif // QUADSTREAM_SIMULATOR_H
