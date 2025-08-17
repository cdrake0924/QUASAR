#ifndef QS_SIMULATOR_H
#define QS_SIMULATOR_H

#include <PostProcessing/ToneMapper.h>
#include <PostProcessing/ShowNormalsEffect.h>

#include <DepthMesh.h>
#include <Quads/FrameGenerator.h>

namespace quasar {

class QSSimulator {
public:
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

    // Reference frame -- QS only has one frame
    std::vector<FrameRenderTarget> refFrameRTs;
    std::vector<QuadMesh> refFrameMeshes;
    std::vector<Node> refFrameNodesRemote;

    // Local objects
    std::vector<Node> refFrameNodesLocal;
    std::vector<Node> refFrameWireframesLocal;

    std::vector<DepthMesh> depthMeshes;
    std::vector<Node> depthNodes;

    std::vector<FrameRenderTarget> copyRTs;

    uint maxViews;

    uint maxVertices = MAX_NUM_PROXIES * NUM_SUB_QUADS * VERTICES_IN_A_QUAD;
    uint maxIndices = MAX_NUM_PROXIES * NUM_SUB_QUADS * INDICES_IN_A_QUAD;
    uint maxVerticesDepth;

    std::vector<std::vector<char>> quads;
    std::vector<std::vector<char>> depthOffsets;

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
        QuadFrame::Sizes sizes;
    } stats;

    QSSimulator(QuadFrame& quadFrame, uint maxViews, FrameGenerator& frameGenerator)
        : quadFrame(quadFrame)
        , frameGenerator(frameGenerator)
        , maxViews(maxViews)
        , quads(maxViews)
        , depthOffsets(maxViews)
        , maxVerticesDepth(quadFrame.getSize().x * quadFrame.getSize().y)
    {
        proxiesPerQuadSet.reserve(maxViews);
        depthOffsetsPerQuadSet.reserve(maxViews);

        refFrameRTs.reserve(maxViews);
        copyRTs.reserve(maxViews);
        refFrameMeshes.reserve(maxViews);
        depthMeshes.reserve(maxViews);
        refFrameNodesLocal.reserve(maxViews);
        refFrameNodesRemote.reserve(maxViews);
        refFrameWireframesLocal.reserve(maxViews);
        depthNodes.reserve(maxViews);

        // Match QuadStream's params:
        auto& quadsGenerator = frameGenerator.quadsGenerator;
        quadsGenerator.params.expandEdges = true;
        quadsGenerator.params.depthThreshold = 1e-4f;
        quadsGenerator.params.flattenThreshold = 0.05f;
        quadsGenerator.params.proxySimilarityThreshold = 0.1f;
        quadsGenerator.params.maxIterForceMerge = 1; // only merge once

        RenderTargetCreateParams rtParams = {
            .width = quadFrame.getSize().x,
            .height = quadFrame.getSize().y,
            .internalFormat = GL_RGBA16F,
            .format = GL_RGBA,
            .type = GL_HALF_FLOAT,
            .wrapS = GL_CLAMP_TO_EDGE,
            .wrapT = GL_CLAMP_TO_EDGE,
            .minFilter = GL_NEAREST,
            .magFilter = GL_NEAREST,
        };
        MeshSizeCreateParams meshParams({
            .maxVertices = maxVertices,
            .maxIndices = maxIndices,
            .vertexSize = sizeof(QuadVertex),
            .attributes = QuadVertex::getVertexInputAttributes(),
            .usage = GL_DYNAMIC_DRAW,
            .indirectDraw = true,
        });
        for (int view = 0; view < maxViews; view++) {
            if (view == maxViews - 1) {
                rtParams.width = 1280; rtParams.height = 720;
            }
            refFrameRTs.emplace_back(rtParams);
            copyRTs.emplace_back(rtParams);

            // We can use less vertices and indicies for the additional views since they will be sparser
            refFrameMeshes.emplace_back(quadFrame, refFrameRTs[view].colorTexture, MAX_NUM_PROXIES / (view == 0 || view != maxViews - 1 ? 1 : 4));
            refFrameNodesLocal.emplace_back(&refFrameMeshes[view]);
            refFrameNodesLocal[view].frustumCulled = false;

            const glm::vec4& color = colors[view % colors.size()];

            refFrameWireframesLocal.emplace_back(&refFrameMeshes[view]);
            refFrameWireframesLocal[view].frustumCulled = false;
            refFrameWireframesLocal[view].wireframe = true;
            refFrameWireframesLocal[view].overrideMaterial = new QuadMaterial({ .baseColor = color });

            depthMeshes.emplace_back(quadFrame.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
            depthNodes.emplace_back(&depthMeshes[view]);
            depthNodes[view].frustumCulled = false;
            depthNodes[view].primativeType = GL_POINTS;

            refFrameNodesRemote.emplace_back(&refFrameMeshes[view]);
            refFrameNodesRemote[view].frustumCulled = false;
            refFrameNodesRemote[view].visible = (view == 0);
            meshScene.addChildNode(&refFrameNodesRemote[view]);
        }
    }
    ~QSSimulator() = default;

    uint getNumTriangles() const {
        uint numTriangles = 0;
        for (const auto& mesh : refFrameMeshes) {
            auto size = mesh.getBufferSizes();
            numTriangles += size.numIndices / 3; // Each triangle has 3 indices
        }
        return numTriangles;
    }

    void addMeshesToScene(Scene& localScene) {
        for (int view = 0; view < maxViews; view++) {
            localScene.addChildNode(&refFrameNodesLocal[view]);
            localScene.addChildNode(&refFrameWireframesLocal[view]);
            localScene.addChildNode(&depthNodes[view]);
        }
    }

    void generateFrame(
        const std::vector<PerspectiveCamera> remoteCameras, Scene& remoteScene,
        DeferredRenderer& remoteRenderer,
        bool showNormals = false, bool showDepth = false)
    {
        // Reset stats
        stats = { 0 };

        for (int view = 0; view < maxViews; view++) {
            auto& remoteCameraToUse = remoteCameras[view];

            auto& gBufferToUse = refFrameRTs[view];

            auto& meshToUse = refFrameMeshes[view];
            auto& meshToUseDepth = depthMeshes[view];

            double startTime = timeutils::getTimeMicros();

            // Center view
            if (view == 0) {
                // Render all objects in remoteScene normally
                remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
            }
            // Other view
            else {
                // Make all previous refFrameMeshes visible and everything else invisible
                for (int prevView = 1; prevView < maxViews; prevView++) {
                    meshScene.rootNode.children[prevView]->visible = (prevView < view);
                }
                // Draw old refFrameMeshes at new remoteCamera view, filling stencil buffer with 1
                remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
                remoteRenderer.pipeline.writeMaskState.disableColorWrites();
                remoteRenderer.drawObjectsNoLighting(meshScene, remoteCameraToUse);

                // Render remoteScene using stencil buffer as a mask
                // At values where stencil buffer is not 1, remoteScene should render
                remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
                remoteRenderer.pipeline.rasterState.polygonOffsetEnabled = false;
                remoteRenderer.pipeline.writeMaskState.enableColorWrites();
                remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                remoteRenderer.pipeline.stencilState.restoreStencilState();
            }
            if (!showNormals) {
                remoteRenderer.copyToFrameRT(gBufferToUse);
                toneMapper.drawToRenderTarget(remoteRenderer, copyRTs[view]);
            }
            else {
                showNormalsEffect.drawToRenderTarget(remoteRenderer, gBufferToUse);
            }
            stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

            auto sizes = frameGenerator.generateRefFrame(
                gBufferToUse, remoteCameraToUse,
                meshToUse
            );
            // QS has data structures that are 103 bits
            stats.sizes.quadsSize += sizes.quadsSize * (103.0) / (8*sizeof(QuadMapDataPacked));
            stats.sizes.depthOffsetsSize += sizes.depthOffsetsSize;
            stats.sizes.numQuads += sizes.numQuads;
            stats.sizes.numDepthOffsets += sizes.numDepthOffsets;

            proxiesPerQuadSet[view] = sizes.numQuads;
            depthOffsetsPerQuadSet[view] = sizes.numDepthOffsets;

            // Copy quads and depth offsets to local vectors
            quadFrame.mapToCPU(quads[view], depthOffsets[view]);

            stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
            stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
            stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
            stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

            stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
            stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
            stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
            stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

            stats.totalCompressTime += frameGenerator.stats.timeToCompress;

            // For debugging: Generate point cloud from depth map
            if (showDepth) {
                meshToUseDepth.update(remoteCameraToUse, gBufferToUse);
                stats.totalGenDepthTime += meshToUseDepth.stats.genDepthTime;
            }
        }
    }

    uint saveToFile(const Path& outputPath) {
        uint totalOutputSize = 0;
        for (int view = 0; view < maxViews; view++) {
            // Save quads
            double startTime = timeutils::getTimeMicros();
            Path filename = (outputPath / "quads").appendToName(std::to_string(view)).withExtension(".bin.zstd");
            std::ofstream quadsFile = std::ofstream(filename, std::ios::binary);
            quadsFile.write(quads[view].data(), quads[view].size());
            quadsFile.close();
            spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                         proxiesPerQuadSet[view], static_cast<double>(quads[view].size()) / BYTES_PER_MEGABYTE,
                            timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

            // Save depth offsets
            startTime = timeutils::getTimeMicros();
            Path offsetsFile = (outputPath / "depthOffsets").appendToName(std::to_string(view)).withExtension(".bin.zstd");
            std::ofstream depthOffsetsFile = std::ofstream(offsetsFile, std::ios::binary);
            depthOffsetsFile.write(depthOffsets[view].data(), depthOffsets[view].size());
            depthOffsetsFile.close();
            spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                         depthOffsetsPerQuadSet[view], static_cast<double>(depthOffsets[view].size()) / BYTES_PER_MEGABYTE,
                            timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

            // Save color buffer
            Path colorFileName = outputPath / ("color" + std::to_string(view));
            copyRTs[view].saveColorAsJPG(colorFileName.withExtension(".jpg"));

            totalOutputSize += quads[view].size() + depthOffsets[view].size();
        }

        return totalOutputSize;
    }

private:
    QuadFrame& quadFrame;

    FrameGenerator& frameGenerator;

    std::vector<uint> proxiesPerQuadSet;
    std::vector<uint> depthOffsetsPerQuadSet;

    // Shaders
    ToneMapper toneMapper;
    ShowNormalsEffect showNormalsEffect;

    Scene meshScene;

    QuadMaterial wireframeMaterial = QuadMaterial({. baseColor = colors[0] });
    QuadMaterial maskWireframeMaterial = QuadMaterial({. baseColor = colors[colors.size()-1] });
};

} // namespace quasar


#endif // QS_SIMULATOR_H
