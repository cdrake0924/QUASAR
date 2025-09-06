#include <spdlog/spdlog.h>
#include <Buffer.h>

using namespace quasar;

Buffer::Buffer() {
    glGenBuffers(1, &ID);
}

Buffer::Buffer(const BufferCreateParams& params)
    : target(params.target)
    , usage(params.usage)
    , numElems(params.numElems)
    , maxElems(params.maxElems)
    , dataSize(params.dataSize)
{
    glGenBuffers(1, &ID);
    bind();
    setData(params.numElems, params.data);
    unbind();
}

Buffer::Buffer(const Buffer& other)
    : target(other.target)
    , usage(other.usage)
    , numElems(other.numElems)
    , maxElems(other.maxElems)
    , dataSize(other.dataSize)
{
    glGenBuffers(1, &ID);
    bind();
    std::vector<char> data(other.numElems * other.dataSize);
#ifdef GL_CORE
    other.getSubData(0, other.numElems, data.data());
#else
    other.getData(data.data());
#endif
    setData(other.numElems, data.data());
    unbind();
}

Buffer::Buffer(Buffer&& other) noexcept
    : target(other.target)
    , usage(other.usage)
    , numElems(other.numElems)
    , maxElems(other.maxElems)
    , dataSize(other.dataSize)
{
    ID = other.ID;
    other.ID = 0;
    other.numElems = 0;
}

Buffer::~Buffer() {
    glDeleteBuffers(1, &ID);
}

Buffer& Buffer::operator=(const Buffer& other) {
    if (this == &other) return *this;

    glDeleteBuffers(1, &ID);

    target = other.target;
    usage = other.usage;
    numElems = other.numElems;
    dataSize = other.dataSize;
    glGenBuffers(1, &ID);

    if (numElems > 0) {
        bind();
        std::vector<char> data(numElems * dataSize);
#ifdef GL_CORE
        other.getSubData(0, numElems, data.data());
#else
        other.getData(data.data());
#endif
        setData(numElems, data.data());
    }

    return *this;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) return *this;

    glDeleteBuffers(1, &ID);

    ID = other.ID;
    target = other.target;
    usage = other.usage;
    numElems = other.numElems;
    dataSize = other.dataSize;

    other.ID = 0;
    other.numElems = 0;

    return *this;
}

void Buffer::bind() const {
    glBindBuffer(target, ID);
}

void Buffer::unbind() const {
    glBindBuffer(target, 0);
}

void Buffer::bindToUniformBlock(GLuint shaderID, const std::string& blockName, GLuint bindingIndex) const {
    GLuint blockIndex = glGetUniformBlockIndex(shaderID, blockName.c_str());
    if (blockIndex == GL_INVALID_INDEX) {
        return;
    }
    glUniformBlockBinding(shaderID, blockIndex, bindingIndex);
    glBindBufferBase(target, bindingIndex, ID);
}

size_t Buffer::getSize() const {
    return numElems;
}

void Buffer::resize(size_t newNumElems, bool copy) {
    if (newNumElems == numElems) {
        return;
    }

    std::vector<char> data;
    if (copy) {
        data.resize(numElems * dataSize);
        getData(data.data());
    }

    glBufferData(target, newNumElems * dataSize, nullptr, usage);

    if (copy) {
        size_t elemsToCopy = std::min(numElems, newNumElems);
        glBufferSubData(target, 0, elemsToCopy * dataSize, data.data());
    }

    numElems = newNumElems;
}

void Buffer::smartResize(size_t newNumElems, bool copy) {
    if (newNumElems == numElems) {
        return;
    }

    if (newNumElems > numElems) {
        size_t targetNumElems = std::max(numElems + numElems / 2, newNumElems); // grow by 1.5x
        targetNumElems = std::min(targetNumElems, maxElems == 0 ? targetNumElems : maxElems);
        resize(targetNumElems, copy);
    }
    else if (newNumElems <= numElems / 4) {
        size_t targetNumElems = std::max(numElems / 2, newNumElems);
        resize(targetNumElems, copy);
    }
}

#ifdef GL_CORE
void Buffer::getSubData(size_t offset, size_t numElems, void* data) const {
    glGetBufferSubData(target, offset * dataSize, numElems * dataSize, data);
}
#endif

void Buffer::getData(void* data) const {
#ifdef GL_CORE
    getSubData(0, numElems, data);
#else
    void* mappedBuffer = glMapBufferRange(target, 0, numElems * dataSize, GL_MAP_READ_BIT);
    if (mappedBuffer) {
        std::memcpy(data, mappedBuffer, numElems * dataSize);
        glUnmapBuffer(target);
    }
    else {
        spdlog::error("Could not map buffer data.");
    }
#endif
}

template<typename T>
std::vector<T> Buffer::getData() const {
    static_assert(std::is_trivially_copyable<T>::value, "Buffer data must be trivially copyable.");
    if (sizeof(T) != dataSize) {
        spdlog::error("Data size mismatch. Requested type has size {}, but buffer holds size {}.", sizeof(T), dataSize);
        return {};
    }
    std::vector<T> data(numElems);
    getData(static_cast<void*>(data.data()));
    return data;
}

void Buffer::setData(size_t numElems, const void* data) {
    resize(numElems);
    glBufferData(target, numElems * dataSize, data, usage);
}

void Buffer::setData(const std::vector<char>& data) {
    setData(data.size() / dataSize, data.data());
}

#ifdef GL_CORE
void Buffer::setSubData(size_t offset, size_t numElems, const void* data) {
    glBufferSubData(target, offset * dataSize, numElems * dataSize, data);
}

void Buffer::setSubData(size_t offset, const std::vector<char>& data) {
    setSubData(offset, data.size() / dataSize, data.data());
}
#endif

void* Buffer::mapToCPU(GLbitfield access) const {
    void* ptr = glMapBufferRange(target, 0, numElems * dataSize, access);
    if (!ptr) {
        spdlog::error("Failed to map buffer to CPU.");
    }
    return ptr;
}

void Buffer::unmapFromCPU() const {
    if (!glUnmapBuffer(target)) {
        spdlog::error("Buffer data became corrupted while mapped.");
    }
}
