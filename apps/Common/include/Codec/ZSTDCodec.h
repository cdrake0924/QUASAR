#ifndef ZSTD_CODEC_H
#define ZSTD_CODEC_H

#include <vector>
#include <thread>

#include <zstd.h>

#include <Codec/Codec.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class ZSTDCodec : public Codec {
public:
    ZSTDCodec(
            uint32_t compressionLevel = ZSTD_CLEVEL_DEFAULT,
            uint32_t compressionStrategy = ZSTD_dfast,
            uint32_t numWorkers = std::thread::hardware_concurrency() / 2,
            uint32_t chunkSize = BYTES_PER_MEGABYTE)
        : compressionCtx(ZSTD_createCCtx())
        , decompressionCtx(ZSTD_createDCtx())
    {
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_compressionLevel, compressionLevel);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_strategy, compressionStrategy);

        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_nbWorkers, numWorkers);
        ZSTD_CCtx_setParameter(compressionCtx, ZSTD_c_jobSize, chunkSize);
    }

    ~ZSTDCodec() override {
        ZSTD_freeCCtx(compressionCtx);
        ZSTD_freeDCtx(decompressionCtx);
    }

    size_t compress(const void* uncompressedData, std::vector<char>& compressedData, size_t numBytesUncompressed) override {
        size_t maxCompressedBytes = ZSTD_compressBound(numBytesUncompressed);
        compressedData.resize(maxCompressedBytes);

        return ZSTD_compress2(
            compressionCtx,
            compressedData.data(), maxCompressedBytes,
            uncompressedData, numBytesUncompressed);
    }

    size_t decompress(const std::vector<char>& compressedData, std::vector<char>& decompressedData) override {
        return ZSTD_decompressDCtx(decompressionCtx,
            decompressedData.data(), decompressedData.size(),
            compressedData.data(), compressedData.size());
    }

private:
    ZSTD_CCtx* compressionCtx;
    ZSTD_DCtx* decompressionCtx;
};

} // namespace quasar

#endif // ZSTD_CODEC_H
