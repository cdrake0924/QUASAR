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
    size_t numElems = 0;
    size_t maxElems = 0;
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

    size_t getSize() const;

    void resize(size_t newNumElems, bool copy = false);
    void smartResize(size_t newNumElems, bool copy = false);

#ifdef GL_CORE
    void getSubData(size_t offset, size_t numElems, void* data) const;
#endif

    void getData(void* data) const;
    template<typename T>
    std::vector<T> getData() const;

    void setData(size_t numElems, const void* data);
    void setData(const std::vector<char>& data);

#ifdef GL_CORE
    void setSubData(size_t offset, size_t numElems, const void* data);
    void setSubData(size_t offset, const std::vector<char>& data);
#endif

    void* mapToCPU(GLbitfield access = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT) const;
    void unmapFromCPU() const;

private:
    GLenum target = GL_ARRAY_BUFFER;
    GLenum usage = GL_STATIC_DRAW;
    size_t numElems;
    size_t dataSize;
    size_t maxElems;
};

} // namespace quasar

#endif // BUFFER_H
