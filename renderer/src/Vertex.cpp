#include <Vertex.h>

using namespace quasar;

uint32_t Vertex::nextID = 0;

Vertex::Vertex(glm::vec3 position)
    : position(position)
{}
Vertex::Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords)
    : position(position), normal(normal), texCoords(texCoords)
{}
Vertex::Vertex(glm::vec3 position, glm::vec3 color, glm::vec3 normal)
    : position(position), color(color), normal(normal)
{}
Vertex::Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords, glm::vec3 tangent, glm::vec3 bitangent)
    : position(position), normal(normal), texCoords(texCoords), tangent(tangent), bitangent(bitangent)
{}
Vertex::Vertex(glm::vec3 position, glm::vec3 normal, glm::vec2 texCoords, glm::vec3 tangent)
    : position(position), normal(normal), texCoords(texCoords), tangent(tangent)
{    bitangent = glm::cross(normal, tangent);
}
