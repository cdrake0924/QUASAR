#ifndef BUFFER_H
#define BUFFER_H

#include <vector>
#include <string>
#include <cstddef>

#include <OpenGLObject.h>

namespace quasar {

struct BufferCreateParams {
    GLenum target;
    size_t dataSize;
    uint numElems = 0;
    GLenum usage = GL_STATIC_DRAW;
    const void* data = nullptr;
};

class Buffer : public OpenGLObject {
public:
    Buffer();
    Buffer(const BufferCreateParams& params);
    Buffer(const Buffer& other);
    Buffer(Buffer&& other) noexcept;
    ~Buffer() override;

    Buffer& operator=(const Buffer& other);
    Buffer& operator=(Buffer&& other) noexcept;

    void bind() const override;
    void unbind() const override;

    void bindToUniformBlock(GLuint shaderID, const std::string& blockName, GLuint bindingIndex) const;

    uint32_t getSize() const;

    void resize(uint newNumElems, bool copy = false);
    void smartResize(uint newNumElems, bool copy = false);

#ifdef GL_CORE
    void getSubData(uint offset, uint numElems, void* data) const;
#endif

    void getData(void* data) const;
    template<typename T>
    std::vector<T> getData() const;

    void setData(uint numElems, const void* data);
    void setData(const std::vector<char>& data);

#ifdef GL_CORE
    void setSubData(uint offset, uint numElems, const void* data);
    void setSubData(uint offset, const std::vector<char>& data);
#endif

    void* mapToCPU(GLbitfield access = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT) const;
    void unmapFromCPU() const;

private:
    GLenum target = GL_ARRAY_BUFFER;
    GLenum usage = GL_STATIC_DRAW;
    uint numElems = 0;
    size_t dataSize = 0;
};

} // namespace quasar

#endif // BUFFER_H
