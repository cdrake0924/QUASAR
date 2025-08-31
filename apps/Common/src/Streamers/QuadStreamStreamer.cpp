#include <Streamers/QuadStreamStreamer.h>

using namespace quasar;

QuadStreamStreamer::QuadStreamStreamer(
        QuadSet& quadSet,
        uint maxViews,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        FrameGenerator& frameGenerator)
    : quadSet(quadSet)
    , maxViews(maxViews)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , frameGenerator(frameGenerator)
    , maxVerticesDepth(quadSet.getSize().x * quadSet.getSize().y)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
{
    referenceFrameRTs.reserve(maxViews);
    copyRTs.reserve(maxViews);
    referenceFrameMeshes.reserve(maxViews);
    depthMeshes.reserve(maxViews);
    referenceFrameNodesLocal.reserve(maxViews);
    referenceFrameNodesRemote.reserve(maxViews);
    referenceFrameWireframesLocal.reserve(maxViews);
    depthNodes.reserve(maxViews);

    referenceFrames.resize(maxViews);

    // Mostly match QuadStream's params from paper:
    auto quadsGenerator = frameGenerator.getQuadsGenerator();
    quadsGenerator->params.expandEdges = true;
    quadsGenerator->params.depthThreshold = 1e-4f;
    quadsGenerator->params.flattenThreshold = 0.05f; // This has been changed from original paper
    quadsGenerator->params.proxySimilarityThreshold = 0.1f;
    quadsGenerator->params.maxIterForceMerge = 1; // Only merge once (similar-ish to doing quad splitting)

    RenderTargetCreateParams rtParams = {
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
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
        referenceFrameRTs.emplace_back(rtParams);
        copyRTs.emplace_back(rtParams);

        // We can use less vertices and indicies for the additional views since they will be sparser
        uint maxProxies = (view == 0 || view == maxViews - 1) ? MAX_QUADS_PER_MESH : MAX_QUADS_PER_MESH / 4;
        referenceFrameMeshes.emplace_back(quadSet, referenceFrameRTs[view].colorTexture, maxProxies);
        referenceFrameNodesLocal.emplace_back(&referenceFrameMeshes[view]);
        referenceFrameNodesLocal[view].frustumCulled = false;

        const glm::vec4& color = colors[view % colors.size()];

        referenceFrameWireframesLocal.emplace_back(&referenceFrameMeshes[view]);
        referenceFrameWireframesLocal[view].frustumCulled = false;
        referenceFrameWireframesLocal[view].wireframe = true;
        referenceFrameWireframesLocal[view].overrideMaterial = new QuadMaterial({ .baseColor = color });

        depthMeshes.emplace_back(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        depthNodes.emplace_back(&depthMeshes[view]);
        depthNodes[view].frustumCulled = false;
        depthNodes[view].primitiveType = GL_POINTS;

        referenceFrameNodesRemote.emplace_back(&referenceFrameMeshes[view]);
        referenceFrameNodesRemote[view].frustumCulled = false;
        referenceFrameNodesRemote[view].visible = (view == 0);
        meshScene.addChildNode(&referenceFrameNodesRemote[view]);
    }
}

uint QuadStreamStreamer::getNumTriangles() const {
    uint numTriangles = 0;
    for (const auto& mesh : referenceFrameMeshes) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}

void QuadStreamStreamer::addMeshesToScene(Scene& localScene) {
    for (int view = 0; view < maxViews; view++) {
        localScene.addChildNode(&referenceFrameNodesLocal[view]);
        localScene.addChildNode(&referenceFrameWireframesLocal[view]);
        localScene.addChildNode(&depthNodes[view]);
    }
}

void QuadStreamStreamer::generateFrame(
    const std::vector<PerspectiveCamera> remoteCameras, Scene& remoteScene,
    DeferredRenderer& remoteRenderer,
    bool showNormals, bool showDepth)
{
    // Reset stats
    stats = { 0 };

    for (int view = 0; view < maxViews; view++) {
        auto& remoteCameraToUse = remoteCameras[view];

        auto& gBufferToUse = referenceFrameRTs[view];

        auto& meshToUse = referenceFrameMeshes[view];
        auto& meshToUseDepth = depthMeshes[view];

        double startTime = timeutils::getTimeMicros();

        // Center view
        if (view == 0) {
            // Render all objects in remoteScene normally
            remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
        }
        // Other view
        else {
            // Make all previous referenceFrameMeshes visible and everything else invisible
            for (int prevView = 1; prevView < maxViews; prevView++) {
                meshScene.rootNode.children[prevView]->visible = (prevView < view);
            }
            // Draw old referenceFrameMeshes at new remoteCamera view, filling stencil buffer with 1
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

        frameGenerator.createReferenceFrame(
            gBufferToUse, remoteCameraToUse,
            meshToUse,
            referenceFrames[view]
        );

        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

        stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totaltimeToCreateMeshMs += frameGenerator.stats.timeToCreateMeshMs;

        stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            meshToUseDepth.update(remoteCameraToUse, gBufferToUse);
            stats.totalGenDepthTime += meshToUseDepth.stats.genDepthTime;
        }

        stats.totalSizes.numQuads += referenceFrames[view].getTotalNumQuads();
        stats.totalSizes.numDepthOffsets += referenceFrames[view].getTotalNumDepthOffsets();
        // QS has data structures that are 103 bits. We approximate their data size by multiplying by 103.
        stats.totalSizes.quadsSize += referenceFrames[view].getTotalQuadsSize() * (103.0) / (8*sizeof(QuadMapDataPacked));
        stats.totalSizes.depthOffsetsSize += referenceFrames[view].getTotalDepthOffsetsSize();
    }
}

size_t QuadStreamStreamer::writeToFile(const Path& outputPath) {
    size_t totalOutputSize = 0;
    for (int view = 0; view < maxViews; view++) {
        // Save color
        Path colorFileName = outputPath / ("color" + std::to_string(view));
        copyRTs[view].writeColorAsJPG(colorFileName.withExtension(".jpg"));

        totalOutputSize += referenceFrames[view].writeToFiles(outputPath, view);
    }
    return totalOutputSize;
}
