#include <QuadsStreamer.h>

using namespace quasar;

QuadsStreamer::QuadsStreamer(
        QuadSet& quadSet,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const std::string& receiverURL)
    : receiverURL(receiverURL)
    , quadSet(quadSet)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , frameGenerator(quadSet)
    , referenceFrameRT({
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
    , residualFrameRT({
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
    , residualFrameMaskRT({
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
    , residualFrameMesh(quadSet, residualFrameRT.colorTexture, MAX_QUADS_PER_MESH / 4)
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
    , DataStreamerTCP(receiverURL)
{
    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    for (int i = 0; i < meshScenes.size(); i++) {
        referenceFrameMeshes.emplace_back(quadSet, referenceFrameRT.colorTexture);

        referenceFrameNodes.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodes[i].frustumCulled = false;
        meshScenes[i].addChildNode(&referenceFrameNodes[i]);

        referenceFrameNodesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameNodesLocal[i].frustumCulled = false;

        referenceFrameWireframesLocal.emplace_back(&referenceFrameMeshes[i]);
        referenceFrameWireframesLocal[i].frustumCulled = false;
        referenceFrameWireframesLocal[i].wireframe = true;
        referenceFrameWireframesLocal[i].visible = false;
        referenceFrameWireframesLocal[i].overrideMaterial = &wireframeMaterial;
    }

    residualFrameNode.setEntity(&residualFrameMesh);
    residualFrameNode.frustumCulled = false;

    residualFrameWireframeNodesLocal.setEntity(&residualFrameMesh);
    residualFrameWireframeNodesLocal.frustumCulled = false;
    residualFrameWireframeNodesLocal.wireframe = true;
    residualFrameWireframeNodesLocal.visible = false;
    residualFrameWireframeNodesLocal.overrideMaterial = &maskWireframeMaterial;

    depthNode.setEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primativeType = GL_POINTS;

    if (!receiverURL.empty()) {
        spdlog::info("Created QuadsStreamer that sends to URL: {}", receiverURL);
    }
}

uint QuadsStreamer::getNumTriangles() const {
    auto refMeshSizes = referenceFrameMeshes[currMeshIndex].getBufferSizes();
    auto maskMeshSizes = residualFrameMesh.getBufferSizes();
    return (refMeshSizes.numIndices + maskMeshSizes.numIndices) / 3; // Each triangle has 3 indices
}

void QuadsStreamer::addMeshesToScene(Scene& localScene) {
    for (int i = 0; i < 2; i++) {
        localScene.addChildNode(&referenceFrameNodesLocal[i]);
        localScene.addChildNode(&referenceFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&residualFrameNode);
    localScene.addChildNode(&residualFrameWireframeNodesLocal);
    localScene.addChildNode(&depthNode);
}

void QuadsStreamer::generateFrame(
    DeferredRenderer& remoteRenderer, Scene& remoteScene, const PerspectiveCamera& remoteCamera,
    bool createResidualFrame, bool showNormals, bool showDepth)
{
    // Reset stats
    stats = { 0 };

    auto& remoteCameraToUse = createResidualFrame ? remoteCameraPrev : remoteCamera;

    // Render remote scene normally
    double startTime = timeutils::getTimeMicros();
    remoteRenderer.drawObjects(remoteScene, remoteCameraToUse);
    if (!showNormals) {
        remoteRenderer.copyToFrameRT(referenceFrameRT);
        toneMapper.drawToRenderTarget(remoteRenderer, copyRT);
    }
    else {
        showNormalsEffect.drawToRenderTarget(remoteRenderer, referenceFrameRT);
    }
    stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Generate Reference Frame
    ============================
    */
    auto quadsGenerator = frameGenerator.getQuadsGenerator();
    quadsGenerator->params.expandEdges = false;
    frameGenerator.createReferenceFrame(
        referenceFrameRT,
        remoteCameraToUse,
        referenceFrameMeshes[currMeshIndex],
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
        quadsGenerator->params.expandEdges = true;
        frameGenerator.updateResidualRenderTargets(
            residualFrameMaskRT, residualFrameRT,
            remoteRenderer, remoteScene,
            meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
            remoteCamera, remoteCameraPrev
        );
        frameGenerator.createResidualFrame(
            residualFrameMaskRT, residualFrameRT,
            remoteCamera, remoteCameraPrev,
            referenceFrameMeshes[currMeshIndex], residualFrameMesh,
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

    residualFrameNode.visible = createResidualFrame;
    currMeshIndex = (currMeshIndex + 1) % meshScenes.size();
    prevMeshIndex = (prevMeshIndex + 1) % meshScenes.size();

    // Only update the previous camera pose if we are not generating a Residual Frame
    if (!createResidualFrame) {
        remoteCameraPrev.setProjectionMatrix(remoteCamera.getProjectionMatrix());
        remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    }

    // For debugging: Generate point cloud from depth map
    if (showDepth) {
        depthMesh.update(remoteCamera, referenceFrameRT);
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

void QuadsStreamer::sendProxies(const PerspectiveCamera& remoteCamera, bool createResidualFrame) {
    if (!receiverURL.empty()) {
        writeToMemory(remoteCamera, createResidualFrame, compressedData);
        send(compressedData);
    }
}

size_t QuadsStreamer::writeToFile(const PerspectiveCamera& remoteCamera, const Path& outputPath) {
    // Save camera
    Path cameraFileName = (outputPath / "camera").withExtension(".bin");
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToFile(cameraFileName);

    // Save color
    Path colorFileName = (outputPath / "color").withExtension(".jpg");
    copyRT.writeColorAsJPG(colorFileName);

    // Save proxies
    return referenceFrame.writeToFiles(outputPath);
}

size_t QuadsStreamer::writeToMemory(const PerspectiveCamera& remoteCamera, bool isResidualFrame, std::vector<char>& outputData) {
    // Save camera data
    std::vector<char> cameraData;
    cameraPose.setProjectionMatrix(remoteCamera.getProjectionMatrix());
    cameraPose.setViewMatrix(remoteCamera.getViewMatrix());
    cameraPose.writeToMemory(cameraData);

    // Save color data
    std::vector<unsigned char> colorData;
    copyRT.writeColorJPGToMemory(colorData);

    // Save geometry data
    std::vector<char> geometryData;
    if (!isResidualFrame) {
        referenceFrame.writeToMemory(geometryData);
    }
    else {
        residualFrame.writeToMemory(geometryData);
    }

    QuadsReceiver::Header header{
        !isResidualFrame ? FrameType::REFERENCE : FrameType::RESIDUAL,
        static_cast<uint32_t>(cameraData.size()),
        static_cast<uint32_t>(colorData.size()),
        static_cast<uint32_t>(geometryData.size())
    };

    spdlog::debug("Writing camera size: {}", header.cameraSize);
    spdlog::debug("Writing color size: {}", header.colorSize);
    spdlog::debug("Writing geometry size: {}", header.geometrySize);

    outputData.resize(sizeof(header) + cameraData.size() + colorData.size() + geometryData.size());

    char* ptr = outputData.data();
    // Write header
    std::memcpy(ptr, &header, sizeof(header));
    ptr += sizeof(header);
    // Write camera data
    std::memcpy(ptr, cameraData.data(), cameraData.size());
    ptr += cameraData.size();
    // Write color data
    std::memcpy(ptr, colorData.data(), colorData.size());
    ptr += colorData.size();
    // Write geometry data
    std::memcpy(ptr, geometryData.data(), geometryData.size());
    ptr += geometryData.size();

    return outputData.size();
}
