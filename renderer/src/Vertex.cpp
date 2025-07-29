#include <Vertex.h>

using namespace quasar;

Vertex::Vertex(const glm::vec3& position)
    : position(position)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec3& normal, const glm::vec2& texCoord)
    : position(position), normal(normal), texCoord(texCoord)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec3& color, const glm::vec3& normal)
    : position(position), color(color), normal(normal)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec3& normal, const glm::vec2& texCoord, const glm::vec3& tangent, const glm::vec3& bitangent)
    : position(position), normal(normal), texCoord(texCoord), tangent(tangent), bitangent(bitangent)
{}
Vertex::Vertex(const glm::vec3& position, const glm::vec3& normal, const glm::vec2& texCoord, const glm::vec3& tangent)
    : position(position), normal(normal), texCoord(texCoord), tangent(tangent)
{
    bitangent = glm::cross(normal, tangent);
}
