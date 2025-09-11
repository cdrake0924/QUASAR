#ifndef QUADS_VERTEX_H
#define QUADS_VERTEX_H

namespace quasar {

struct QuadVertex {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 texCoord3D;
    alignas(16) glm::vec2 expandedUV;

    static const VertexInputAttributes getVertexInputAttributes() {
        return {
            {0, 3, GL_FLOAT, GL_FALSE, offsetof(QuadVertex, position)},
            {1, 3, GL_FLOAT, GL_FALSE, offsetof(QuadVertex, texCoord3D)},
            {2, 2, GL_FLOAT, GL_FALSE, offsetof(QuadVertex, expandedUV)},
        };
    }
};

} // namespace quasar

#endif // QUADS_VERTEX_H
