#include <algorithm>
#include <Quads/FrameGenerator.h>

using namespace quasar;

FrameGenerator::FrameGenerator(QuadSet& quadSet, DeferredRenderer& remoteRenderer, Scene& remoteScene)
    : quadSet(quadSet)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , quadsGenerator(quadSet)
    , maskedRT({
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
{}

void FrameGenerator::generateRefFrame(
    const FrameRenderTarget& referenceFrameRT, const PerspectiveCamera& remoteCamera,
    QuadMesh& mesh, ReferenceFrame& resultFrame,
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

void FrameGenerator::generateResFrame(
    Scene& currMeshScene, Scene& prevMeshScene,
    FrameRenderTarget& residualFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
    QuadMesh& currMesh, QuadMesh& maskMesh, ResidualFrame& resultFrame,
    bool compress)
{
    stats = {};

    const glm::vec2 gBufferSize = glm::vec2(residualFrameRT.width, residualFrameRT.height);
    if (residualFrameRT.width != maskedRT.width || residualFrameRT.height != maskedRT.height) {
        maskedRT.resize(gBufferSize.x, gBufferSize.y);
    }

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
        remoteRenderer.copyToFrameRT(maskedRT); // Save result into a temporary render target
    }
    double timeToRenderUpdated = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Create proxies and meshes for the updated/masked portion of the residual frame
    startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(maskedRT, prevRemoteCamera);
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
    currMesh.appendQuads(quadSet, gBufferSize, false);
    currMesh.createMeshFromProxies(quadSet, gBufferSize, prevRemoteCamera);
    stats.timeToAppendQuadsMs = currMesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = currMesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = currMesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Generate frame from new camera pose using current frame as a mask to capture disocclusions due to camera movement
    startTime = timeutils::getTimeMicros();
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
        remoteRenderer.copyToFrameRT(residualFrameRT);
    }
    stats.timeToRenderMaskMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime) + timeToRenderUpdated;

    // Create proxies and meshes for the revealed portion of the residual frame
    startTime = timeutils::getTimeMicros();
    quadsGenerator.createProxiesFromRT(residualFrameRT, currRemoteCamera);
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

    // Wait for compression to finish and set resulting data sizes
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
