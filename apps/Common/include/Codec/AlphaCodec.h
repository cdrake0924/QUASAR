#ifndef ALPHA_CODEC_H
#define ALPHA_CODEC_H

#include <Codec/Codec.h>

namespace quasar {

class AlphaCodec {
public:
    AlphaCodec(uint width, uint height)
        : width(width)
        , height(height)
    {}
    ~AlphaCodec() = default;

    size_t compress(std::vector<unsigned char>& uncompressedData) {
        return uncompressedData.size();
    }

    size_t decompress(std::vector<unsigned char>& compressedData) {
        return compressedData.size();
    }

private:
    uint width, height;
};

} // namespace quasar

#endif // ALPHA_CODEC_H
