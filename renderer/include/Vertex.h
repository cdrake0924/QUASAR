#ifndef VERTEX_H
#define VERTEX_H

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <OpenGLObject.h>

namespace quasar {

struct VertexInputAttribute {
    GLuint index;
    GLint size;
    GLenum type;
    GLboolean normalized;
    GLintptr pointer;
};
typedef std::vector<VertexInputAttribute> VertexInputAttributes;

struct Vertex {
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 color = glm::vec3(1.0f);
    alignas(16) glm::vec3 normal;
    alignas(16) glm::vec2 texCoords;
    alignas(16) glm::vec3 tangent;
    alignas(16) glm::vec3 bitangent;

    bool operator==(const Vertex& other) const {
        return position == other.position && normal == other.normal && texCoords == other.texCoords &&
               tangent == other.tangent && bitangent == other.bitangent;
    }

    Vertex() = default;
    Vertex(glm::vec3 position);
    Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords);
    Vertex(glm::vec3 position, glm::vec3 color, glm::vec3 normal);
    Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords, glm::vec3 tangent, glm::vec3 bitangent);
    Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords, glm::vec3 tangent);

    static uint32_t nextID;

    static const VertexInputAttributes getVertexInputAttributes() {
        return {
            {0, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, position)},
            {1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, color)},
            {2, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, normal)},
            {3, 2, GL_FLOAT, GL_FALSE, offsetof(Vertex, texCoords)},
            {4, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, tangent)},
            {5, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, bitangent)},
        };
    }
};

} // namespace quasar

namespace std {
    template<> struct hash<quasar::Vertex> {
        size_t operator()(quasar::Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^
                   (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoords) << 1 ^
                   (hash<glm::vec3>()(vertex.tangent) << 1) >> 1) ^
                   (hash<glm::vec3>()(vertex.bitangent) << 1 >> 1);
        }
    };
}

#endif // VERTEX_H
