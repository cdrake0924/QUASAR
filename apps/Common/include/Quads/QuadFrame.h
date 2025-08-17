#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

#include <future>

#include <Utils/FileIO.h>

#include <Codec/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

class QuadFrame {
public:
    struct Stats {
        double timeToCompressMs = 0.0;
        double timeToDecompressMs = 0.0;
    } stats;

    struct Sizes {
        size_t numQuads = 0;
        size_t numDepthOffsets = 0;
        double quadsSize = 0;
        double depthOffsetsSize = 0;

        Sizes& operator+=(const Sizes& other) {
            numQuads += other.numQuads;
            numDepthOffsets += other.numDepthOffsets;
            quadsSize += other.quadsSize;
            depthOffsetsSize += other.depthOffsetsSize;
            return *this;
        }
    };

    // GPU buffers
    QuadBuffers quadBuffers;
    DepthOffsets depthOffsets;

    QuadFrame(const glm::uvec2& frameSize, bool compressed = true)
        : frameSize(frameSize)
        , compress(compressed)
        , quadBuffers(frameSize.x * frameSize.y)
        , depthOffsets(2u * frameSize) // 4 sets of offsets per every pixel (2x2 subpixels)
        , decompressedQuads(sizeof(uint) + quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
        , decompressedDepthOffsets(sizeof(uint) + depthOffsets.size.x * depthOffsets.size.y * 4 * sizeof(uint16_t))
    {}

    const glm::uvec2& getSize() const {
        return frameSize;
    }

    const std::vector<char>& getQuads() const {
        return compressedQuads;
    }

    const std::vector<char>& getDepthOffsets() const {
        return compressedDepthOffsets;
    }

    uint getNumQuads() const {
        return quadBuffers.numProxies;
    }

    uint getNumDepthOffsets() const {
        return depthOffsets.size.x * depthOffsets.size.y;
    }

    void setNumQuads(int numProxies) {
        copiedToCPU = false;
        quadBuffers.resize(numProxies);
    }

#ifdef GL_CORE
    Sizes mapToCPU() {
#if defined(HAS_CUDA)
        // Already copied to CPU, no need to recopy
        if (copiedToCPU) {
            return {
                quadBuffers.numProxies,
                depthOffsets.size.x * depthOffsets.size.y,
                static_cast<double>(compressedQuads.size()),
                static_cast<double>(compressedDepthOffsets.size())
            };
        }

        double startTime = timeutils::getTimeMicros();

        // Copy to CPU
        quadBuffers.mapToCPU(decompressedQuads);
        depthOffsets.mapToCPU(decompressedDepthOffsets);

        std::future<size_t> quadsSizeFuture;
        std::future<size_t> offsetsSizeFuture;

        // compress quads (async)
        if (compress) {
            quadsSizeFuture = quadsCodec.compressAsync(
                decompressedQuads.data(), compressedQuads, decompressedQuads.size());
        }
        else {
            compressedQuads = decompressedQuads;
        }

        // Compress depth offsets (async)
        if (compress) {
            offsetsSizeFuture = depthOffsetsCodec.compressAsync(
                decompressedDepthOffsets.data(), compressedDepthOffsets, decompressedDepthOffsets.size());
        }
        else {
            compressedDepthOffsets = decompressedDepthOffsets;
        }

        // Wait for async results
        if (compress) {
            size_t savedQuadsSize = quadsSizeFuture.get();
            compressedQuads.resize(savedQuadsSize);

            size_t savedDepthOffsetsSize = offsetsSizeFuture.get();
            compressedDepthOffsets.resize(savedDepthOffsetsSize);
        }
        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        copiedToCPU = true;
        return {
            quadBuffers.numProxies,
            depthOffsets.size.x * depthOffsets.size.y,
            static_cast<double>(compressedQuads.size()),
            static_cast<double>(compressedDepthOffsets.size())
        };
#else
        return {
            quadBuffers.numProxies,
            depthOffsets.size.x * depthOffsets.size.y,
            static_cast<double>(compressedQuads.size()),
            static_cast<double>(compressedDepthOffsets.size())
        };
#endif
    }

    Sizes mapToCPU(std::vector<char>& outputQuads, std::vector<char>& outputDepthOffsets) {
        mapToCPU();

        outputQuads.resize(compressedQuads.size());
        outputDepthOffsets.resize(compressedDepthOffsets.size());

        std::copy(compressedQuads.begin(), compressedQuads.end(), outputQuads.begin());
        std::copy(compressedDepthOffsets.begin(), compressedDepthOffsets.end(), outputDepthOffsets.begin());

        return {
            quadBuffers.numProxies,
            depthOffsets.size.x * depthOffsets.size.y,
            static_cast<double>(compressedQuads.size()),
            static_cast<double>(compressedDepthOffsets.size())
        };
    }
#endif

    Sizes loadFromFiles(const std::string& quadsFilename, const std::string& depthOffsetsFilename) {
        double startTime = timeutils::getTimeMicros();

        // Load quads
        compressedQuads = FileIO::loadBinaryFile(quadsFilename);
        quadsCodec.decompress(compressedQuads, decompressedQuads);
        quadBuffers.unmapFromCPU(decompressedQuads);

        // Load depth offsets
        compressedDepthOffsets = FileIO::loadBinaryFile(depthOffsetsFilename);
        depthOffsetsCodec.decompress(compressedDepthOffsets, decompressedDepthOffsets);
        depthOffsets.unmapFromCPU(decompressedDepthOffsets);

        stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        copiedToCPU = true;
        return {
            quadBuffers.numProxies,
            depthOffsets.size.x * depthOffsets.size.y,
            static_cast<double>(compressedQuads.size()),
            static_cast<double>(compressedDepthOffsets.size())
        };
    }

private:
    bool compress = true;
    bool copiedToCPU = false;

    glm::uvec2 frameSize;

    ZSTDCodec quadsCodec;
    ZSTDCodec depthOffsetsCodec;

    // CPU buffers
    std::vector<char> compressedQuads;
    std::vector<char> compressedDepthOffsets;

    std::vector<char> decompressedQuads;
    std::vector<char> decompressedDepthOffsets;
};

} // namespace quasar

#endif // QUAD_FRAME_H
