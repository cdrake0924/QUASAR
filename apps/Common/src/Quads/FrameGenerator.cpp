#include <Quads/FrameGenerator.h>

using namespace quasar;

FrameGenerator::FrameGenerator(QuadFrame& quadFrame, DeferredRenderer& remoteRenderer, Scene& remoteScene)
    : quadFrame(quadFrame)
    , remoteRenderer(remoteRenderer)
    , remoteScene(remoteScene)
    , quadsGenerator(quadFrame)
{}

QuadFrame::Sizes FrameGenerator::generateRefFrame(
    const FrameRenderTarget& frameRT,
    const PerspectiveCamera& remoteCamera,
    QuadMesh& mesh)
{
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);

    double startTime = timeutils::getTimeMicros();

    // Create proxies from the current FrameRenderTarget
    quadsGenerator.createProxiesFromRT(frameRT, remoteCamera);
    stats.timeToGenerateQuadsMs = quadsGenerator.stats.timeToGenerateQuadsMs;
    stats.timeToSimplifyQuadsMs = quadsGenerator.stats.timeToSimplifyQuadsMs;
    stats.timeToGatherQuadsMs = quadsGenerator.stats.timeToGatherQuadsMs;
    stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    auto sizes = quadFrame.mapToCPU();
    stats.timeToCompress = quadFrame.stats.timeToCompressMs;

    // Create mesh from the proxies
    startTime = timeutils::getTimeMicros();
    {
        mesh.appendQuads(quadFrame, gBufferSize);
        mesh.createMeshFromProxies(quadFrame, gBufferSize, remoteCamera);
    }
    stats.timeToAppendQuadsMs = mesh.stats.timeToAppendQuadsMs;
    stats.timeToFillQuadIndicesMs = mesh.stats.timeToGatherQuadsMs;
    stats.timeToCreateVertIndMs = mesh.stats.timeToCreateMeshMs;
    stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    return sizes;
}

QuadFrame::Sizes FrameGenerator::generateResFrame(
    Scene& currScene, Scene& prevScene,
    FrameRenderTarget& frameRT, FrameRenderTarget& maskFrameRT,
    const PerspectiveCamera& currRemoteCamera, const PerspectiveCamera& prevRemoteCamera,
    QuadMesh& currMesh, QuadMesh& maskMesh)
{
    const glm::vec2 gBufferSize = glm::vec2(frameRT.width, frameRT.height);
    QuadFrame::Sizes outputSizes;

    double startTime = timeutils::getTimeMicros();

    // Generate frame using previous frame as a mask for animations
    {
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(prevScene, prevRemoteCamera);

        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.depthState.depthFunc = GL_EQUAL;
        remoteRenderer.drawObjectsNoLighting(currScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.depthState.depthFunc = GL_LESS;
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, prevRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(frameRT);
    }

    // Generate frame using current frame as a mask for movement
    {
        remoteRenderer.pipeline.stencilState.enableRenderingIntoStencilBuffer(GL_KEEP, GL_KEEP, GL_REPLACE);
        remoteRenderer.pipeline.writeMaskState.disableColorWrites();
        remoteRenderer.drawObjectsNoLighting(currScene, currRemoteCamera);

        remoteRenderer.pipeline.stencilState.enableRenderingUsingStencilBufferAsMask(GL_NOTEQUAL, 1);
        remoteRenderer.pipeline.writeMaskState.enableColorWrites();
        remoteRenderer.drawObjects(remoteScene, currRemoteCamera, GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        remoteRenderer.pipeline.stencilState.restoreStencilState();
        remoteRenderer.copyToFrameRT(maskFrameRT);
    }

    stats.timeToRenderMaskMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

    // Create proxies and meshes for the updated keyframe
    {
        startTime = timeutils::getTimeMicros();
        quadsGenerator.createProxiesFromRT(frameRT, prevRemoteCamera);
        stats.timeToCreateQuadsMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto sizes = quadFrame.mapToCPU();
        outputSizes.numQuads = sizes.numQuads;
        outputSizes.numDepthOffsets = sizes.numDepthOffsets;
        outputSizes.quadsSize = sizes.quadsSize;
        outputSizes.depthOffsetsSize = sizes.depthOffsetsSize;
        stats.timeToCompress = quadFrame.stats.timeToCompressMs;

        startTime = timeutils::getTimeMicros();
        {
            currMesh.appendQuads(quadFrame, gBufferSize, false);
            currMesh.createMeshFromProxies(quadFrame, gBufferSize, prevRemoteCamera);
        }
        stats.timeToAppendQuadsMs = currMesh.stats.timeToAppendQuadsMs;
        stats.timeToFillQuadIndicesMs = currMesh.stats.timeToGatherQuadsMs;
        stats.timeToCreateVertIndMs = currMesh.stats.timeToCreateMeshMs;
        stats.timeToCreateMeshMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    // Create proxies and meshes for the updated mask
    {
        startTime = timeutils::getTimeMicros();
        quadsGenerator.createProxiesFromRT(maskFrameRT, currRemoteCamera);
        stats.timeToCreateQuadsMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        auto sizes = quadFrame.mapToCPU();
        outputSizes.numQuads += sizes.numQuads;
        outputSizes.numDepthOffsets += sizes.numDepthOffsets;
        outputSizes.quadsSize += sizes.quadsSize;
        outputSizes.depthOffsetsSize += sizes.depthOffsetsSize;
        stats.timeToCompress = quadFrame.stats.timeToCompressMs;

        startTime = timeutils::getTimeMicros();
        {
            maskMesh.appendQuads(quadFrame, gBufferSize);
            maskMesh.createMeshFromProxies(quadFrame, gBufferSize, currRemoteCamera);
        }
        stats.timeToAppendQuadsMs += maskMesh.stats.timeToAppendQuadsMs;
        stats.timeToFillQuadIndicesMs += maskMesh.stats.timeToGatherQuadsMs;
        stats.timeToCreateVertIndMs += maskMesh.stats.timeToCreateMeshMs;
        stats.timeToCreateMeshMs += timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
    }

    return outputSizes;
}
