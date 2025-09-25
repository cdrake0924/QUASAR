#ifndef ALPHA_CODEC_H
#define ALPHA_CODEC_H

#include <Codec/Codec.h>

namespace quasar {

class AlphaCodec {
public:
    struct Stats {
        double compressTimeMs = 0.0;
        double decompressTimeMs = 0.0;
    } stats;

    AlphaCodec(uint width, uint height)
        : width(width)
        , height(height)
    {}
    ~AlphaCodec() = default;

    size_t compress(std::vector<unsigned char>& uncompressedData) {
        double startTime = timeutils::getTimeMicros();
        size_t res = uncompressedData.size();
        stats.compressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return res;
    }

    size_t decompress(std::vector<unsigned char>& compressedData) {
        double startTime = timeutils::getTimeMicros();
        size_t res = compressedData.size();
        stats.decompressTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);
        return res;
    }

private:
    uint width, height;
};

} // namespace quasar

#endif // ALPHA_CODEC_H
