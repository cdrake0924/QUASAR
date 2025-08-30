#include <QUASARStreamer.h>

using namespace quasar;

QUASARStreamer::QUASARStreamer(
        QuadSet& quadSet,
        uint maxLayers,
        DeferredRenderer& remoteRenderer,
        Scene& remoteScene,
        const PerspectiveCamera& remoteCamera,
        FrameGenerator& frameGenerator)
    : quadSet(quadSet)
    , maxLayers(maxLayers)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , frameGenerator(frameGenerator)
    , maxVerticesDepth(quadSet.getSize().x * quadSet.getSize().y)
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
    , depthMesh(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f))
    // We can use less vertices and indicies for the mask since it will be sparse
    , residualFrameMesh(quadSet, residualFrameRT.colorTexture, MAX_QUADS_PER_MESH / 4)
    , wireframeMaterial({ .baseColor = colors[0] })
    , maskWireframeMaterial({ .baseColor = colors[colors.size()-1] })
{
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());
    remoteCameraPrev.setViewMatrix(remoteCamera.getViewMatrix());

    meshScenes.resize(2);
    referenceFrameMeshes.reserve(meshScenes.size());
    referenceFrameNodes.reserve(meshScenes.size());
    wideFovNodes.reserve(meshScenes.size());
    referenceFrameNodesLocal.reserve(meshScenes.size());
    referenceFrameWireframesLocal.reserve(meshScenes.size());

    copyRTs.reserve(maxLayers);
    referenceFrames.resize(maxLayers);
    residualFrames.resize(maxLayers);

    uint numHidLayers = maxLayers - 1;

    frameRTsHidLayer.reserve(numHidLayers);
    meshesHidLayer.reserve(numHidLayers);
    depthMeshsHidLayer.reserve(numHidLayers);
    nodesHidLayer.reserve(numHidLayers);
    wireframesHidLayer.reserve(numHidLayers);
    depthNodesHidLayer.reserve(numHidLayers);

    // Setup visible layer for reference frame
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

    // Setup masks for residual frame
    residualFrameNode.setEntity(&residualFrameMesh);
    residualFrameNode.frustumCulled = false;

    residualFrameWireframesLocal.setEntity(&residualFrameMesh);
    residualFrameWireframesLocal.frustumCulled = false;
    residualFrameWireframesLocal.wireframe = true;
    residualFrameWireframesLocal.visible = false;
    residualFrameWireframesLocal.overrideMaterial = &maskWireframeMaterial;

    // Setup depth mesh
    depthNode.setEntity(&depthMesh);
    depthNode.frustumCulled = false;
    depthNode.visible = false;
    depthNode.primitiveType = GL_POINTS;

    // Setup hidden layers and wide fov RTs
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
    copyRTs.emplace_back(rtParams);
    for (int layer = 0; layer < numHidLayers; layer++) {
        copyRTs.emplace_back(rtParams);
        frameRTsHidLayer.emplace_back(rtParams);
    }

    for (int layer = 0; layer < numHidLayers; layer++) {
        // We can use less vertices and indicies for the hidden layers since they will be sparser
        meshesHidLayer.emplace_back(quadSet, frameRTsHidLayer[layer].colorTexture, MAX_QUADS_PER_MESH / 4);

        nodesHidLayer.emplace_back(&meshesHidLayer[layer]);
        nodesHidLayer[layer].frustumCulled = false;

        const glm::vec4& color = colors[(layer + 1) % colors.size()];

        wireframesHidLayer.emplace_back(&meshesHidLayer[layer]);
        wireframesHidLayer[layer].frustumCulled = false;
        wireframesHidLayer[layer].wireframe = true;
        wireframesHidLayer[layer].overrideMaterial = new QuadMaterial({ .baseColor = color });

        depthMeshsHidLayer.emplace_back(quadSet.getSize(), glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        depthNodesHidLayer.emplace_back(&depthMeshsHidLayer[layer]);
        depthNodesHidLayer[layer].frustumCulled = false;
        depthNodesHidLayer[layer].primitiveType = GL_POINTS;
    }

    // Setup scene to use as mask for wide fov camera
    for (int i = 0; i < meshScenes.size(); i++) {
        wideFovNodes.emplace_back(&referenceFrameMeshes[i]);
        wideFovNodes[i].frustumCulled = false;
        sceneWideFov.addChildNode(&wideFovNodes[i]);
    }
    for (int i = 0; i < numHidLayers - 1; i++) {
        sceneWideFov.addChildNode(&nodesHidLayer[i]);
    }
    sceneWideFov.addChildNode(&residualFrameNode);
}

uint QUASARStreamer::getNumTriangles() const {
    auto refMeshSizes = referenceFrameMeshes[currMeshIndex].getBufferSizes();
    uint numTriangles = refMeshSizes.numIndices / 3; // Each triangle has 3 indices
    for (const auto& mesh : meshesHidLayer) {
        auto size = mesh.getBufferSizes();
        numTriangles += size.numIndices / 3; // Each triangle has 3 indices
    }
    return numTriangles;
}

void QUASARStreamer::addMeshesToScene(Scene& localScene) {
    for (int i = 0; i < meshScenes.size(); i++) {
        localScene.addChildNode(&referenceFrameNodesLocal[i]);
        localScene.addChildNode(&referenceFrameWireframesLocal[i]);
    }
    localScene.addChildNode(&residualFrameNode);
    localScene.addChildNode(&residualFrameWireframesLocal);
    localScene.addChildNode(&depthNode);

    for (int layer = 0; layer < maxLayers - 1; layer++) {
        localScene.addChildNode(&nodesHidLayer[layer]);
        localScene.addChildNode(&wireframesHidLayer[layer]);
        localScene.addChildNode(&depthNodesHidLayer[layer]);
    }
}

void QUASARStreamer::generateFrame(
    const PerspectiveCamera& remoteCameraCenter, const PerspectiveCamera& remoteCameraWideFov, Scene& remoteScene,
    DeferredRenderer& remoteRenderer,
    DepthPeelingRenderer& remoteRendererDP,
    bool createResidualFrame, bool showNormals, bool showDepth)
{
    // Reset stats
    stats = { 0 };

    // Render remote scene with multiple layers
    double startTime = timeutils::getTimeMicros();
    remoteRendererDP.drawObjects(remoteScene, remoteCameraCenter);
    stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    for (int layer = 0; layer < maxLayers; layer++) {
        int hiddenIndex = layer - 1;
        auto& remoteCameraToUse = (layer == 0 && createResidualFrame) ? remoteCameraPrev :
                                    ((layer != maxLayers - 1) ? remoteCameraCenter : remoteCameraWideFov);

        auto& frameToUse = (layer == 0) ? referenceFrameRT : frameRTsHidLayer[hiddenIndex];

        auto& meshToUse = (layer == 0) ? referenceFrameMeshes[currMeshIndex] : meshesHidLayer[hiddenIndex];
        auto& meshToUseDepth = (layer == 0) ? depthMesh : depthMeshsHidLayer[hiddenIndex];

        startTime = timeutils::getTimeMicros();
        if (layer == 0) {
            remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse);
            if (!showNormals) {
                remoteRenderer.copyToFrameRT(frameToUse);
                toneMapper.drawToRenderTarget(remoteRenderer, copyRTs[layer]);
            }
            else {
                showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse);
            }
        }
        else if (layer != maxLayers - 1) {
            // Copy to render target
            if (!showNormals) {
                remoteRendererDP.peelingLayers[hiddenIndex+1].blitToFrameRT(frameToUse);
                toneMapper.setUniforms(frameToUse);
                toneMapper.drawToRenderTarget(remoteRendererDP, copyRTs[layer], false);
            }
            else {
                showNormalsEffect.drawToRenderTarget(remoteRendererDP, frameToUse);
            }
        }
        // Wide fov camera
        else {
            // Draw old center mesh at new remoteCamera layer, filling stencil buffer with 1
            remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
            remoteRenderer.pipeline.writeMaskState.disableColorWrites();
            wideFovNodes[currMeshIndex].visible = false;
            wideFovNodes[prevMeshIndex].visible = true;
            remoteRenderer.drawObjectsNoLighting(sceneWideFov, remoteCameraToUse);

            // Render remoteScene using stencil buffer as a mask
            // At values where stencil buffer is not 1, remoteScene should render
            remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
            remoteRenderer.pipeline.writeMaskState.enableColorWrites();
            remoteRenderer.drawObjectsNoLighting(remoteScene, remoteCameraToUse, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            remoteRenderer.pipeline.stencilState.restoreStencilState();

            if (!showNormals) {
                remoteRenderer.copyToFrameRT(frameToUse);
                toneMapper.setUniforms(frameToUse);
                toneMapper.drawToRenderTarget(remoteRenderer, copyRTs[layer], false);
            }
            else {
                showNormalsEffect.drawToRenderTarget(remoteRenderer, frameToUse);
            }
        }
        stats.totalRenderTime += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        /*
        ============================
        Generate Reference Frame
        ============================
        */
        auto quadsGenerator = frameGenerator.getQuadsGenerator();
        auto oldParams = quadsGenerator->params;
        if (layer == maxLayers - 1) {
            quadsGenerator->params.maxIterForceMerge = 4;
            quadsGenerator->params.depthThreshold = 1e-3f;
            quadsGenerator->params.flattenThreshold = 0.5f;
            quadsGenerator->params.proxySimilarityThreshold = 5.0f;
        }
        else if (layer > 0) {
            quadsGenerator->params.maxIterForceMerge = 4;
            quadsGenerator->params.depthThreshold = 1e-3f;
        }
        quadsGenerator->params.expandEdges = false;
        frameGenerator.createReferenceFrame(
            frameToUse, remoteCameraToUse,
            meshToUse,
            referenceFrames[layer],
            !createResidualFrame // No need to waste time compressing if we are generating a residual frame
        );

        quadsGenerator->params = oldParams;

        stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
        stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
        stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
        stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

        stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
        stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToFillQuadIndicesMs;
        stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
        stats.totaltimeToCreateMeshMs += frameGenerator.stats.timeToCreateMeshMs;

        if (layer != 0 || !createResidualFrame) {
            stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
        }

        /*
        ============================
        Generate Residual Frame
        ============================
        */
        if (layer == 0) {
            if (createResidualFrame) {
                quadsGenerator->params.expandEdges = true;
                frameGenerator.updateResidualRenderTargets(
                    residualFrameMaskRT, residualFrameRT,
                    remoteRenderer, remoteScene,
                    meshScenes[currMeshIndex], meshScenes[prevMeshIndex],
                    remoteCameraCenter, remoteCameraPrev
                );
                frameGenerator.createResidualFrame(
                    residualFrameMaskRT, residualFrameRT,
                    remoteCameraCenter, remoteCameraPrev,
                    referenceFrameMeshes[currMeshIndex], residualFrameMesh,
                    residualFrames[layer]
                );

                stats.totalRenderTime += frameGenerator.stats.timeToUpdateRTsMs;

                stats.totalGenQuadMapTime += frameGenerator.stats.timeToGenerateQuadsMs;
                stats.totalSimplifyTime += frameGenerator.stats.timeToSimplifyQuadsMs;
                stats.totalGatherQuadsTime += frameGenerator.stats.timeToGatherQuadsMs;
                stats.totalCreateProxiesTime += frameGenerator.stats.timeToCreateQuadsMs;

                stats.totalAppendQuadsTime += frameGenerator.stats.timeToAppendQuadsMs;
                stats.totalFillQuadsIndiciesTime += frameGenerator.stats.timeToGatherQuadsMs;
                stats.totalCreateVertIndTime += frameGenerator.stats.timeToCreateVertIndMs;
                stats.totaltimeToCreateMeshMs += frameGenerator.stats.timeToCreateMeshMs;

                stats.totalCompressTime += frameGenerator.stats.timeToCompressMs;
            }

            residualFrameNode.visible = createResidualFrame;
            currMeshIndex = (currMeshIndex + 1) % meshScenes.size();
            prevMeshIndex = (prevMeshIndex + 1) % meshScenes.size();

            // Only update the previous camera pose if we are not generating a Residual Frame
            if (!createResidualFrame) {
                remoteCameraPrev.setViewMatrix(remoteCameraCenter.getViewMatrix());
            }
        }

        // For debugging: Generate point cloud from depth map
        if (showDepth) {
            meshToUseDepth.update(remoteCameraToUse, frameToUse);
            stats.totalGenDepthTime += meshToUseDepth.stats.genDepthTime;
        }

        if (!createResidualFrame) {
            stats.totalSizes.numQuads += referenceFrames[layer].getTotalNumQuads();
            stats.totalSizes.numDepthOffsets += referenceFrames[layer].getTotalNumDepthOffsets();
            stats.totalSizes.quadsSize += referenceFrames[layer].getTotalQuadsSize();
            stats.totalSizes.depthOffsetsSize += referenceFrames[layer].getTotalDepthOffsetsSize();
        }
        else {
            stats.totalSizes.numQuads += residualFrames[layer].getTotalNumQuads();
            stats.totalSizes.numDepthOffsets += residualFrames[layer].getTotalNumDepthOffsets();
            stats.totalSizes.quadsSize += residualFrames[layer].getTotalQuadsSize();
            stats.totalSizes.depthOffsetsSize += residualFrames[layer].getTotalDepthOffsetsSize();
        }
    }
}

size_t QUASARStreamer::writeToFile(const Path& outputPath) {
    size_t totalOutputSize = 0;
    for (int layer = 0; layer < maxLayers; layer++) {
        // Save color
        Path colorFileName = outputPath / ("color" + std::to_string(layer));
        copyRTs[layer].writeColorAsJPG(colorFileName.withExtension(".jpg"));

        // Save proxies
        totalOutputSize += referenceFrames[layer].writeToFiles(outputPath, layer);
    }
    return totalOutputSize;
}
