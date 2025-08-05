#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

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
        uint numQuads = 0;
        uint numDepthOffsets = 0;
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
        , depthOffsets(2u * frameSize) // 4 offsets per pixel
        , decompressedQuads(sizeof(uint) + quadBuffers.maxProxies * sizeof(QuadMapDataPacked))
        , decompressedDepthOffsets(sizeof(uint) + depthOffsets.size.x * depthOffsets.size.y * 4 * sizeof(uint16_t))
    {}

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

    const glm::uvec2& getSize() const {
        return frameSize;
    }

    void setNumQuads(int numProxies) {
        copiedToCPU = false;
        quadBuffers.resize(numProxies);
    }

#ifdef GL_CORE
    Sizes copyToMemory() {
        if (copiedToCPU) {
            return { quadBuffers.numProxies, depthOffsets.size.x * depthOffsets.size.y,
                     static_cast<double>(compressedQuads.size()), static_cast<double>(compressedDepthOffsets.size()) };
        }

        double startTime = timeutils::getTimeMicros();

        // Copy quads
        quadBuffers.copyToMemory(decompressedQuads);

        if (compress) {
            uint savedQuadsSize = codec.compress(decompressedQuads.data(), compressedQuads, decompressedQuads.size());
            compressedQuads.resize(savedQuadsSize);
        } else {
            compressedQuads = decompressedQuads;
        }

        // Copy depth offsets
        depthOffsets.copyToMemory(decompressedDepthOffsets);

        if (compress) {
            uint savedDepthOffsetsSize = codec.compress(decompressedDepthOffsets.data(), compressedDepthOffsets, decompressedDepthOffsets.size());
            compressedDepthOffsets.resize(savedDepthOffsetsSize);
        } else {
            compressedDepthOffsets = decompressedDepthOffsets;
        }

        stats.timeToCompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        return { quadBuffers.numProxies, depthOffsets.size.x * depthOffsets.size.y,
                 static_cast<double>(compressedQuads.size()), static_cast<double>(compressedDepthOffsets.size()) };
    }

    Sizes copyToMemory(std::vector<char>& outputQuads, std::vector<char>& outputDepthOffsets) {
        copyToMemory();

        outputQuads.resize(compressedQuads.size());
        outputDepthOffsets.resize(compressedDepthOffsets.size());

        std::copy(compressedQuads.begin(), compressedQuads.end(), outputQuads.begin());
        std::copy(compressedDepthOffsets.begin(), compressedDepthOffsets.end(), outputDepthOffsets.begin());

        return { quadBuffers.numProxies, depthOffsets.size.x * depthOffsets.size.y,
                 static_cast<double>(compressedQuads.size()), static_cast<double>(compressedDepthOffsets.size()) };
    }
#endif

    Sizes loadFromFiles(const std::string& quadsFilename, const std::string& depthOffsetsFilename) {
        double startTime = timeutils::getTimeMicros();

        // Load quads
        compressedQuads = FileIO::loadBinaryFile(quadsFilename);
        codec.decompress(compressedQuads, decompressedQuads);
        quadBuffers.loadFromMemory(decompressedQuads);

        // Load depth offsets
        compressedDepthOffsets = FileIO::loadBinaryFile(depthOffsetsFilename);
        codec.decompress(compressedDepthOffsets, decompressedDepthOffsets);
        depthOffsets.loadFromMemory(decompressedDepthOffsets);

        stats.timeToDecompressMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTime);

        copiedToCPU = true;
        return { quadBuffers.numProxies, depthOffsets.size.x * depthOffsets.size.y,
                 static_cast<double>(compressedQuads.size()), static_cast<double>(compressedDepthOffsets.size()) };
    }

private:
    bool copiedToCPU = false;
    bool compress = true;

    glm::uvec2 frameSize;

    ZSTDCodec codec;

    // CPU buffers
    std::vector<char> compressedQuads;
    std::vector<char> compressedDepthOffsets;

    std::vector<char> decompressedQuads;
    std::vector<char> decompressedDepthOffsets;
};

} // namespace quasar

#endif // QUAD_FRAME_H
