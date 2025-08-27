#ifndef QUAD_MESH_H
#define QUAD_MESH_H

#include <Texture.h>
#include <Primitives/Mesh.h>
#include <Cameras/PerspectiveCamera.h>
#include <Shaders/ComputeShader.h>

#include <Quads/QuadSet.h>
#include <Quads/QuadVertex.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

#define MAX_QUADS_PER_MESH 550000

#define VERTICES_IN_A_QUAD 4
#define INDICES_IN_A_QUAD 6
#define NUM_SUB_QUADS 4

class QuadMesh : public Mesh {
public:
    struct BufferSizes {
        uint numVertices;
        uint numIndices;
    };

    struct Stats {
        double timeToAppendQuadsMs = 0.0;
        double timeToGatherQuadsMs = 0.0;
        double timeToCreateMeshMs = 0.0;
    } stats;

    uint maxProxies;

    QuadMesh(const QuadSet& quadSet, Texture& colorTexture, uint maxQuadsPerMesh = MAX_QUADS_PER_MESH);
    ~QuadMesh() = default;

    void appendQuads(const QuadSet& quadSet, const glm::vec2& gBufferSize, bool isReferenceFrame = true);
    void createMeshFromProxies(const QuadSet& quadSet, const glm::vec2& gBufferSize, const PerspectiveCamera& remoteCamera);

    BufferSizes getBufferSizes() const;

private:
    QuadBuffers currentQuadBuffers;

    Buffer meshSizesBuffer;
    Buffer prevNumProxiesBuffer;
    Buffer currNumProxiesBuffer;

    Buffer quadCreatedFlagsBuffer;
    Buffer quadIndicesMap;

    ComputeShader appendQuadsShader;
    ComputeShader fillQuadIndicesShader;
    ComputeShader createQuadMeshShader;

    void fillQuadIndices(const QuadSet& quadSet, const glm::vec2& gBufferSize);
};

} // namespace quasar

#endif // QUAD_MESH_H
