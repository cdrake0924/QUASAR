#include <future>
#include <algorithm>
#include <Quads/FrameGenerator.h>

using namespace quasar;

FrameGenerator::FrameGenerator(QuadSet& quadSet)
    : quadSet(quadSet)
    , quadsGenerator(quadSet)
{}

void FrameGenerator::createReferenceFrame(
    const FrameRenderTarget& referenceFrameRT, const PerspectiveCamera& remoteCamera,
    QuadMesh& mesh,
    ReferenceFrame& resultFrame,
    bool compress)
{
    stats = {};

    const glm::vec2 gBufferSize = glm::vec2(referenceFrameRT.width, referenceFrameRT.height);

    // Create proxies from the current FrameRenderTarget
    double startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(referenceFrameRT, remoteCamera);
    stats.timeToGenerateQuadsMs = quadsGenerator.stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuadsMs = quadsGenerator.stats.timeToSimplifyQuadsMs;
    stats.timeToGatherQuadsMs = quadsGenerator.stats.timeToGatherQuadsMs;
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizes = quadSet.mapToCPU(uncompressedQuads, uncompressedOffsets);
    resultFrame.numQuads = sizes.numQuads;
    resultFrame.numDepthOffsets = sizes.numDepthOffsets;
    stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

    // Compress proxies (nonblocking)
    std::future<size_t> quadsFuture, offsetsFuture;
    if (compress) {
        quadsFuture = refQuadsCodec.compressAsync(
            uncompressedQuads.data(),
            resultFrame.quads,
            uncompressedQuads.size());
        offsetsFuture = refOffsetsCodec.compressAsync(
            uncompressedOffsets.data(),
            resultFrame.depthOffsets,
            uncompressedOffsets.size());
    }

    // Using GPU buffers, create mesh from proxies
    startTime = timeutils::getTimeMicros();
    mesh.appendQuads(quadSet, gBufferSize);
    mesh.createMeshFromProxies(quadSet, gBufferSize, remoteCamera);
    stats.timeToAppendQuadsMs = mesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = mesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = mesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Wait for compression to finish and set resulting data sizes
    if (compress) {
        resultFrame.quads.resize(quadsFuture.get());
        resultFrame.depthOffsets.resize(offsetsFuture.get());
        stats.timeToCompressMs = std::max(refQuadsCodec.stats.timeToCompressMs, refOffsetsCodec.stats.timeToCompressMs);
    }
}

void FrameGenerator::updateResidualRenderTargets(
    FrameRenderTarget& resFrameMaskRT, FrameRenderTarget& resFrameRT,
    DeferredRenderer& remoteRenderer, Scene& remoteScene,
    Scene& currMeshScene, Scene& prevMeshScene,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera)
{
    stats = {};

    const glm::vec2 gBufferSize = glm::vec2(resFrameRT.width, resFrameRT.height);

    // Generate frame from old camera pose using previous frame as a mask to capture scene changes
    double startTime = timeutils::getTimeMicros();
    {
        // Fill depth buffer with previous generated mesh
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(prevMeshScene, prevRemoteCamera);

        // Use current generated mesh as a stencil mask
        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
        remoteRenderer.drawObjectsNoLighting(currMeshScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // Render scene using stencil mask; this lets only content that is different pass
        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(resFrameMaskRT); // Save result into a temporary render target
    }

    // Generate frame from new camera pose using current frame as a mask to capture disocclusions due to camera movement
    {
        // Use current generated mesh as a stencil mask
        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(currMeshScene, currRemoteCamera);

        // Render scene using stencil mask; this lets only content that is different pass
        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, currRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(resFrameRT);
    }
    stats.timeToRenderMaskMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
}

void FrameGenerator::createResidualFrame(
    const FrameRenderTarget& resFrameMaskRT, const FrameRenderTarget& resFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
    QuadMesh& mesh, QuadMesh& maskMesh,
    ResidualFrame& resultFrame,
    bool compress)
{
    const glm::vec2 gBufferSize = glm::vec2(resFrameRT.width, resFrameRT.height);

    /*
    ============================
    Create proxies and meshes for the masked portion of the residual frame capturing updated geometry
    ============================
    */
    double startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(resFrameMaskRT, prevRemoteCamera);
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer updated proxies to CPU for compression
    auto sizesUpdated = quadSet.mapToCPU(uncompressedQuadsUpdated, uncompressedOffsetsUpdated);
    resultFrame.numQuadsUpdated = sizesUpdated.numQuads;
    resultFrame.numDepthOffsetsUpdated = sizesUpdated.numDepthOffsets;
    stats.timeToTransferMs = quadSet.stats.timeToTransferMs;

    // Compress proxies (asynchronous)
    std::future<size_t> quadsUpdatedFuture, offsetsUpdatedFuture;
    if (compress) {
        quadsUpdatedFuture = resQuadsUpdatedCodec.compressAsync(
            uncompressedQuadsUpdated.data(),
            resultFrame.quadsUpdated,
            uncompressedQuadsUpdated.size());
        offsetsUpdatedFuture = resOffsetsUpdatedCodec.compressAsync(
            uncompressedOffsetsUpdated.data(),
            resultFrame.depthOffsetsUpdated,
            uncompressedOffsetsUpdated.size());
    }

    // Using GPU buffers, create mesh using proxies
    startTime = timeutils::getTimeMicros();
    mesh.appendQuads(quadSet, gBufferSize, false);
    mesh.createMeshFromProxies(quadSet, gBufferSize, prevRemoteCamera);
    stats.timeToAppendQuadsMs = mesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = mesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = mesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Create proxies and meshes for the revealed portion of the residual frame capturing disocclusions
    ============================
    */
    startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(resFrameRT, currRemoteCamera);
    stats.timeToCreateQuadsMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Transfer revealed proxies to CPU for compression
    auto sizesRevealed = quadSet.mapToCPU(uncompressedQuadsRevealed, uncompressedOffsetsRevealed);
    resultFrame.numQuadsRevealed = sizesRevealed.numQuads;
    resultFrame.numDepthOffsetsRevealed = sizesRevealed.numDepthOffsets;
    stats.timeToTransferMs += quadSet.stats.timeToTransferMs;

    // Compress proxies (asynchronous)
    std::future<size_t> quadsRevealedFuture, offsetsRevealedFuture;
    if (compress) {
        quadsRevealedFuture = resQuadsRevealedCodec.compressAsync(
            uncompressedQuadsRevealed.data(),
            resultFrame.quadsRevealed,
            uncompressedQuadsRevealed.size());
        offsetsRevealedFuture = resOffsetsRevealedCodec.compressAsync(
            uncompressedOffsetsRevealed.data(),
            resultFrame.depthOffsetsRevealed,
            uncompressedOffsetsRevealed.size());
    }

    // Using GPU buffers, create mesh using proxies
    startTime = timeutils::getTimeMicros();
    maskMesh.appendQuads(quadSet, gBufferSize);
    maskMesh.createMeshFromProxies(quadSet, gBufferSize, currRemoteCamera);
    stats.timeToAppendQuadsMs += maskMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs += maskMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs += maskMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    /*
    ============================
    Wait for asynchronous compression to finish and set resulting data sizes
    ============================
    */
    if (compress) {
        resultFrame.quadsUpdated.resize(quadsUpdatedFuture.get());
        resultFrame.depthOffsetsUpdated.resize(offsetsUpdatedFuture.get());
        resultFrame.quadsRevealed.resize(quadsRevealedFuture.get());
        resultFrame.depthOffsetsRevealed.resize(offsetsRevealedFuture.get());
        stats.timeToCompressMs = std::max(
            std::max(resQuadsUpdatedCodec.stats.timeToCompressMs, resOffsetsUpdatedCodec.stats.timeToCompressMs),
            std::max(resQuadsRevealedCodec.stats.timeToCompressMs, resOffsetsRevealedCodec.stats.timeToCompressMs)
        );
    }
}
