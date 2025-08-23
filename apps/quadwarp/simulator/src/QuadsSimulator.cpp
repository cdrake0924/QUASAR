#include <QuadsSimulator.h>

using namespace quasar;

QuadsSimulator::QuadsSimulator(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        FrameGenerator& frameGenerator)
    : quadSet(quadSet)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , frameGenerator(frameGenerator)
    , refFrameRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , resFrameRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , resFrameMaskRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    , copyRT({
        .width = quadSet.getSize().x,
        .height = quadSet.getSize().y,
        .internalFormat = GL_RGBA16F,
        .format = GL_RGBA,
        .type = GL_HALF_FLOAT,
        .wrapS = GL_CLAMP_TO_EDGE,
        .wrapT = GL_CLAMP_TO_EDGE,
        .minFilter = GL_NEAREST,
        .magFilter = GL_NEAREST,
    })
    // We can use less vertices and indicies for the mask since it will be sparse
    , resFrameMesh(quadSet, resFrameRT.colorTexture, MAX_NUM_PROXIES / 4)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
{
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    meshScenes.resize(2);
    refFrameMeshes.reserve(meshScenes.size());
    refFrameNodes.reserve(meshScenes.size());
    refFrameNodesLocal.reserve(meshScenes.size());
    refFrameWireframesLocal.reserve(meshScenes.size());

    for (int i = 0; i < meshScenes.size(); i++) {
        refFrameMeshes.emplace_back(quadSet, refFrameRT.colorTexture);

        refFrameNodes.emplace_back(&refFrameMeshes[i]);
        refFrameNodes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&refFrameNodes[i]);

        refFrameNodesLocal.emplace_back(&refFrameMeshes[i]);
        refFrameNodesLocal[i].frustumCulled = false;

        refFrameWireframesLocal.emplace_back(&refFrameMeshes[i]);
        refFrameWireframesLocal[i].frustumCulled = false;
        refFrameWireframesLocal[i].wireframe = true;
        refFrameWireframesLocal[i].visible = false;
        refFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
    }

    resFrameNode.setEntity(&resFrameMesh);
    resFrameNode.frustumCulled = false;

    resFrameWireframeNodesLocal.setEntity(&resFrameMesh);
    resFrameWireframeNodesLocal.frustumCulled = false;
    resFrameWireframeNodesLocal.wireframe = true;
    resFrameWireframeNodesLocal.visible = false;
    resFrameWireframeNodesLocal.overrideMaterial = &maskWireframeMaterial;

    depthNode.setEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primativeType = GL_POINTS;
}

uint QuadsSimulator::getNumTriangles() const {
    auto refMeshSizes = refFrameMeshes[currMeshIndex].getBufferSizes();
    auto maskMeshSizes = resFrameMesh.getBufferSizes();
    return (refMeshSizes.numIndices + maskMeshSizes.numIndices) / 3; // Each triangle has 3 indices
}

void QuadsSimulator::addMeshesToScene(Scene& localScene) {
    for (int i = 0; i < 2; i++) {
        localScene.addChildNode(&refFrameNodesLocal[i]);
        localScene.addChildNode(&refFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&resFrameNode);
    localScene.addChildNode(&resFrameWireframeNodesLocal);
    localScene.addChildNode(&depthNode);
}

void QuadsSimulator::generateFrame(
    const PerspectiveCamera& remoteCamera, Scene& remoteScene,
    DeferredRenderer& remoteRenderer,
    bool createResidualFrame, bool showNormals, bool showDepth)
{
    auto& remoteCameraToUse = createResidualFrame ? remoteCameraPrev : remoteCamera;

    // Reset stats
    stats = { 0 };

    // Render remote scene normally
    double startTime = timeutils::getTimeMicros();
    remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
    if (!showNormals) {
        remoteRenderer.copyToFrameRT(refFrameRT);
        toneMapper.drawToRenderTarget(remoteRenderer, copyRT);
    }
    else {
        showNormalsEffect.drawToRenderTarget(remoteRenderer, refFrameRT);
    }
    stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Generate Reference Frame
    ============================
    */
    auto& quadsGenerator = frameGenerator.quadsGenerator;
    quadsGenerator.params.expandEdges = false;
    frameGenerator.createReferenceFrame(
        refFrameRT,
        remoteCameraToUse,
        refFrameMeshes[currMeshIndex],
        referenceFrame,
        !createResidualFrame // No need to waste time compressing if we are generating a residual frame
    );

    stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
    stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
    stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
    stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

    stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
    stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
    stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
    stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

    stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;

    /*
    ============================
    Generate Residual Frame
    ============================
    */
    if (createResidualFrame) {
        quadsGenerator.params.expandEdges = true;
        frameGenerator.updateResidualRenderTargets(
            resFrameMaskRT, resFrameRT,
            remoteRenderer, remoteScene,
            meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
            remoteCamera, remoteCameraPrev
        );
        frameGenerator.createResidualFrame(
            resFrameMaskRT, resFrameRT,
            remoteCamera, remoteCameraPrev,
            refFrameMeshes[currMeshIndex], resFrameMesh,
            residualFrame
        );

        stats.totalRenderTime += frameGenerator.stats.timeToUpdateRTsMs;

        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

        stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totalCreateMeshTime += frameGenerator.stats.timeToCreateMeshMs;

        stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
    }

    resFrameNode.visible = createResidualFrame;
    currMeshIndex = (currMeshIndex + 1) % meshScenes.size();
    prevMeshIndex = (prevMeshIndex + 1) % meshScenes.size();

    // Only update the previous camera pose if we are not generating a Residual Frame
    if (!createResidualFrame) {
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }

    // For debugging: Generate point cloud from depth map
    if (showDepth) {
        depthMesh.update(remoteCamera, refFrameRT);
        stats.totalGenDepthTime += depthMesh.stats.genDepthTime;
    }

    if (!createResidualFrame) {
        stats.totalSizes.numQuads += referenceFrame.getTotalNumQuads();
        stats.totalSizes.numDepthOffsets += referenceFrame.getTotalNumDepthOffsets();
        stats.totalSizes.quadsSize += referenceFrame.getTotalQuadsSize();
        stats.totalSizes.depthOffsetsSize += referenceFrame.getTotalDepthOffsetsSize();
    }
    else {
        stats.totalSizes.numQuads += residualFrame.getTotalNumQuads();
        stats.totalSizes.numDepthOffsets += residualFrame.getTotalNumDepthOffsets();
        stats.totalSizes.quadsSize += residualFrame.getTotalQuadsSize();
        stats.totalSizes.depthOffsetsSize += residualFrame.getTotalDepthOffsetsSize();
    }
}

size_t QuadsSimulator::saveToFile(const Path& outputPath) {
    // Save color
    Path colorFileName = outputPath / "color";
    copyRT.saveColorAsJPG(colorFileName.withExtension(".jpg"));

    // Save proxies
    return referenceFrame.saveToFiles(outputPath);
}
