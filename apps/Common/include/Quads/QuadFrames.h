#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

#include <future>
#include <algorithm>

#include <spdlog/spdlog.h>

#include <Path.h>
#include <Utils/FileIO.h>
#include <Codec/ZSTDCodec.h>
#include <Quads/QuadBuffers.h>
#include <Quads/DepthOffsets.h>
#include <Utils/TimeUtils.h>

namespace quasar {

enum FrameType {
    NONE,
    REFERENCE,
    RESIDUAL,
};

class QuadFrame {};

class ReferenceFrame : public QuadFrame {
public:
    struct Header {
        uint32_t quadsSize;
        uint32_t depthOffsetsSize;
    };

    size_t numQuads, numDepthOffsets;
    std::vector<char> quads, depthOffsets;

    size_t getTotalNumQuads() const {
        return numQuads;
    }
    size_t getTotalNumDepthOffsets() const {
        return numDepthOffsets;
    }
    double getTotalQuadsSize() const {
        return quads.size();
    }
    double getTotalDepthOffsetsSize() const {
        return depthOffsets.size();
    }

    std::future<size_t> compressAndStoreQuads(const std::vector<char>& uncompressedQuads) {
        auto quadsFuture = refQuadsCodec.compressAsync(
            uncompressedQuads.data(),
            quads,
            uncompressedQuads.size());
        return quadsFuture;
    }
    std::future<size_t> compressAndStoreDepthOffsets(const std::vector<char>& uncompressedOffsets) {
        auto offsetsFuture = refOffsetsCodec.compressAsync(
            uncompressedOffsets.data(),
            depthOffsets,
            uncompressedOffsets.size());
        return offsetsFuture;
    }

    std::future<size_t> decompressQuads(std::vector<char>& outputQuads) {
        auto quadsFuture = refQuadsCodec.decompressAsync(quads, outputQuads);
        return quadsFuture;
    }
    std::future<size_t> decompressDepthOffsets(std::vector<char>& outputOffsets) {
        auto offsetsFuture = refOffsetsCodec.decompressAsync(depthOffsets, outputOffsets);
        return offsetsFuture;
    }

    double getTimeToCompress() const {
        return std::max(refQuadsCodec.stats.timeToCompressMs, refOffsetsCodec.stats.timeToCompressMs);
    }
    double getTimeToDecompress() const {
        return std::max(refQuadsCodec.stats.timeToDecompressMs, refOffsetsCodec.stats.timeToDecompressMs);
    }

    size_t writeToFiles(const Path& outputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Save quads
        double startTime = timeutils::getTimeMicros();
        Path quadsFileName = (outputPath / ("quads" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(quadsFileName, quads.data(), quads.size());
        spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                       numQuads, static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save depth offsets
        startTime = timeutils::getTimeMicros();
        Path depthOffsetsFileName = (outputPath / ("depthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(depthOffsetsFileName, depthOffsets.data(), depthOffsets.size());
        spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsets, static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }

    size_t writeToMemory(std::vector<char>& outputData) {
        Header header{
            static_cast<uint32_t>(quads.size()),
            static_cast<uint32_t>(depthOffsets.size())
        };
        size_t outputSize = sizeof(header) + quads.size() + depthOffsets.size();
        outputData.resize(outputSize);

        char* ptr = outputData.data();
        // Write header
        std::memcpy(ptr, &header, sizeof(Header));
        ptr += sizeof(Header);
        // Write quads
        std::memcpy(ptr, quads.data(), quads.size());
        ptr += quads.size();
        // Write depth offsets
        std::memcpy(ptr, depthOffsets.data(), depthOffsets.size());
        ptr += depthOffsets.size();

        return outputSize;
    }

    size_t loadFromFiles(const Path& inputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Load quads
        double startTime = timeutils::getTimeMicros();
        Path quadsFileName = (inputPath / ("quads" + idxStr)).withExtension(".bin.zstd");
        auto quadsData = FileIO::loadFromBinaryFile(quadsFileName);
        quads = std::move(quadsData);
        spdlog::info("Loaded quads ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        startTime = timeutils::getTimeMicros();
        Path depthOffsetsFileName = (inputPath / ("depthOffsets" + idxStr)).withExtension(".bin.zstd");
        auto depthOffsetsData = FileIO::loadFromBinaryFile(depthOffsetsFileName);
        depthOffsets = std::move(depthOffsetsData);
        spdlog::info("Loaded depth offsets ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }

    size_t loadFromMemory(const std::vector<char>& inputData) {
        const char* ptr = inputData.data();

        // Read header
        Header header;
        std::memcpy(&header, ptr, sizeof(Header));
        ptr += sizeof(Header);

        // Sanity check
        if (inputData.size() < sizeof(Header) + header.quadsSize + header.depthOffsetsSize) {
            throw std::runtime_error("Input data size " +
                                      std::to_string(inputData.size()) +
                                      " is smaller than expected from header " +
                                      std::to_string(header.quadsSize + header.depthOffsetsSize));
        }

        // Read quads
        quads.resize(header.quadsSize);
        std::memcpy(quads.data(), ptr, header.quadsSize);
        ptr += header.quadsSize;

        // Read depth offsets
        depthOffsets.resize(header.depthOffsetsSize);
        std::memcpy(depthOffsets.data(), ptr, header.depthOffsetsSize);
        ptr += header.depthOffsetsSize;

        return quads.size() + depthOffsets.size();
    }

private:
    ZSTDCodec refQuadsCodec;
    ZSTDCodec refOffsetsCodec;
};

class ResidualFrame : public QuadFrame {
public:
    struct Header {
        uint32_t quadsUpdatedSize;
        uint32_t depthOffsetsUpdatedSize;
        uint32_t quadsRevealedSize;
        uint32_t depthOffsetsRevealedSize;
    };

    size_t numQuadsUpdated;
    size_t numDepthOffsetsUpdated;
    size_t numQuadsRevealed;
    size_t numDepthOffsetsRevealed;
    std::vector<char> quadsUpdated;
    std::vector<char> depthOffsetsUpdated;
    std::vector<char> quadsRevealed;
    std::vector<char> depthOffsetsRevealed;

    size_t getTotalNumQuads() const {
        return numQuadsUpdated + numQuadsRevealed;
    }
    size_t getTotalNumDepthOffsets() const {
        return numDepthOffsetsUpdated + numDepthOffsetsRevealed;
    }
    double getTotalQuadsSize() const {
        return quadsUpdated.size() + depthOffsetsUpdated.size();
    }
    double getTotalDepthOffsetsSize() const {
        return quadsRevealed.size() + depthOffsetsRevealed.size();
    }

    std::future<size_t> compressAndStoreUpdatedQuads(const std::vector<char>& uncompressedQuads) {
        auto quadsFuture = resQuadsUpdatedCodec.compressAsync(
            uncompressedQuads.data(),
            quadsUpdated,
            uncompressedQuads.size());
        return quadsFuture;
    }
    std::future<size_t> compressAndStoreRevealedQuads(const std::vector<char>& uncompressedQuads) {
        auto quadsFuture = resQuadsRevealedCodec.compressAsync(
            uncompressedQuads.data(),
            quadsRevealed,
            uncompressedQuads.size());
        return quadsFuture;
    }
    std::future<size_t> compressAndStoreUpdatedDepthOffsets(const std::vector<char>& uncompressedOffsets) {
        auto offsetsFuture = resOffsetsUpdatedCodec.compressAsync(
            uncompressedOffsets.data(),
            depthOffsetsUpdated,
            uncompressedOffsets.size());
        return offsetsFuture;
    }
    std::future<size_t> compressAndStoreRevealedDepthOffsets(const std::vector<char>& uncompressedOffsets) {
        auto offsetsFuture = resOffsetsRevealedCodec.compressAsync(
            uncompressedOffsets.data(),
            depthOffsetsRevealed,
            uncompressedOffsets.size());
        return offsetsFuture;
    }

    std::future<size_t> decompressUpdatedQuads(std::vector<char>& outputQuads) {
        auto quadsFuture = resQuadsUpdatedCodec.decompressAsync(quadsUpdated, outputQuads);
        return quadsFuture;
    }
    std::future<size_t> decompressRevealedQuads(std::vector<char>& outputQuads) {
        auto quadsFuture = resQuadsRevealedCodec.decompressAsync(quadsRevealed, outputQuads);
        return quadsFuture;
    }
    std::future<size_t> decompressUpdatedDepthOffsets(std::vector<char>& outputOffsets) {
        auto offsetsFuture = resOffsetsUpdatedCodec.decompressAsync(depthOffsetsUpdated, outputOffsets);
        return offsetsFuture;
    }
    std::future<size_t> decompressRevealedDepthOffsets(std::vector<char>& outputOffsets) {
        auto offsetsFuture = resOffsetsRevealedCodec.decompressAsync(depthOffsetsRevealed, outputOffsets);
        return offsetsFuture;
    }

    double getTimeToCompress() const {
        return std::max(
            std::max(resQuadsUpdatedCodec.stats.timeToCompressMs, resOffsetsUpdatedCodec.stats.timeToCompressMs),
            std::max(resQuadsRevealedCodec.stats.timeToCompressMs, resOffsetsRevealedCodec.stats.timeToCompressMs)
        );
    }
    double getTimeToDecompress() const {
        return std::max(
            std::max(resQuadsUpdatedCodec.stats.timeToDecompressMs, resOffsetsUpdatedCodec.stats.timeToDecompressMs),
            std::max(resQuadsRevealedCodec.stats.timeToDecompressMs, resOffsetsRevealedCodec.stats.timeToDecompressMs)
        );
    }

    size_t writeToFiles(const Path& outputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Save updated quads
        double startTime = timeutils::getTimeMicros();
        Path updatedQuadsFileName = (outputPath / ("updatedQuads" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(updatedQuadsFileName, quadsUpdated.data(), quadsUpdated.size());
        spdlog::info("Saved {} updated quads ({:.3f}MB) in {:.3f}ms",
                       numQuadsUpdated, static_cast<double>(quadsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save updated depth offsets
        startTime = timeutils::getTimeMicros();
        Path updatedDepthOffsetsFileName = (outputPath / ("updatedDepthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(updatedDepthOffsetsFileName, depthOffsetsUpdated.data(), depthOffsetsUpdated.size());
        spdlog::info("Saved {} updated depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsUpdated, static_cast<double>(depthOffsetsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed quads
        startTime = timeutils::getTimeMicros();
        Path revealedQuadsFileName = (outputPath / ("revealedQuads" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(revealedQuadsFileName, quadsRevealed.data(), quadsRevealed.size());
        spdlog::info("Saved {} revealed quads ({:.3f}MB) in {:.3f}ms",
                        numQuadsRevealed, static_cast<double>(quadsRevealed.size()) / BYTES_PER_MEGABYTE,
                          timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed depth offsets
        startTime = timeutils::getTimeMicros();
        Path revealedDepthOffsetsFileName = (outputPath / ("revealedDepthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::writeToBinaryFile(revealedDepthOffsetsFileName, depthOffsetsRevealed.data(), depthOffsetsRevealed.size());
        spdlog::info("Saved {} revealed depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsRevealed, static_cast<double>(depthOffsetsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quadsUpdated.size() + depthOffsetsUpdated.size() +
               quadsRevealed.size() + depthOffsetsRevealed.size();
    }

    size_t loadFromFiles(const Path& inputPath) {
        // Load updated quads
        double startTime = timeutils::getTimeMicros();
        Path updatedQuadsFileName = (inputPath / "updatedQuads").withExtension(".bin.zstd");
        auto quadsUpdatedData = FileIO::loadFromBinaryFile(updatedQuadsFileName);
        quadsUpdated = std::move(quadsUpdatedData);
        spdlog::info("Loaded {} updated quads ({:.3f}MB) in {:.3f}ms",
                       numQuadsUpdated, static_cast<double>(quadsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Load updated depth offsets
        startTime = timeutils::getTimeMicros();
        Path updatedDepthOffsetsFileName = (inputPath / "updatedDepthOffsets").withExtension(".bin.zstd");
        auto depthOffsetsUpdatedData = FileIO::loadFromBinaryFile(updatedDepthOffsetsFileName);
        depthOffsetsUpdated = std::move(depthOffsetsUpdatedData);
        spdlog::info("Loaded {} updated depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsUpdated, static_cast<double>(depthOffsetsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Load revealed quads
        startTime = timeutils::getTimeMicros();
        Path revealedQuadsFileName = (inputPath / "revealedQuads").withExtension(".bin.zstd");
        auto quadsRevealedData = FileIO::loadFromBinaryFile(revealedQuadsFileName);
        quadsRevealed = std::move(quadsRevealedData);
        spdlog::info("Loaded {} revealed quads ({:.3f}MB) in {:.3f}ms",
                       numQuadsRevealed, static_cast<double>(quadsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Load revealed depth offsets
        startTime = timeutils::getTimeMicros();
        Path revealedDepthOffsetsFileName = (inputPath / "revealedDepthOffsets").withExtension(".bin.zstd");
        auto depthOffsetsRevealedData = FileIO::loadFromBinaryFile(revealedDepthOffsetsFileName);
        depthOffsetsRevealed = std::move(depthOffsetsRevealedData);
        spdlog::info("Loaded {} revealed depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsRevealed, static_cast<double>(depthOffsetsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quadsUpdated.size() + depthOffsetsUpdated.size() +
               quadsRevealed.size() + depthOffsetsRevealed.size();
    }

    size_t writeToMemory(std::vector<char>& outputData) {
        Header header{
            static_cast<uint32_t>(quadsUpdated.size()),
            static_cast<uint32_t>(depthOffsetsUpdated.size()),
            static_cast<uint32_t>(quadsRevealed.size()),
            static_cast<uint32_t>(depthOffsetsRevealed.size())
        };
        size_t outputSize = sizeof(Header) +
                            quadsUpdated.size() + depthOffsetsUpdated.size() +
                            quadsRevealed.size() + depthOffsetsRevealed.size();
        outputData.resize(outputSize);

        char* ptr = outputData.data();
        // Write header
        std::memcpy(ptr, &header, sizeof(Header));
        ptr += sizeof(Header);
        // Write updated quads
        std::memcpy(ptr, quadsUpdated.data(), quadsUpdated.size());
        ptr += quadsUpdated.size();
        // Write updated depth offsets
        std::memcpy(ptr, depthOffsetsUpdated.data(), depthOffsetsUpdated.size());
        ptr += depthOffsetsUpdated.size();
        // Write revealed quads
        std::memcpy(ptr, quadsRevealed.data(), quadsRevealed.size());
        ptr += quadsRevealed.size();
        // Write revealed depth offsets
        std::memcpy(ptr, depthOffsetsRevealed.data(), depthOffsetsRevealed.size());
        ptr += depthOffsetsRevealed.size();

        return outputData.size();
    }

    size_t loadFromMemory(const std::vector<char>& inputData) {
        const char* ptr = inputData.data();

        // Read header
        Header header;
        std::memcpy(&header, ptr, sizeof(Header));
        ptr += sizeof(Header);

        // Sanity check
        if (inputData.size() < sizeof(Header) +
                               header.quadsUpdatedSize + header.depthOffsetsUpdatedSize +
                               header.quadsRevealedSize + header.depthOffsetsRevealedSize) {
            throw std::runtime_error("Input data size " +
                                      std::to_string(inputData.size()) +
                                      " is smaller than expected from header " +
                                      std::to_string(header.quadsUpdatedSize + header.depthOffsetsUpdatedSize +
                                                     header.quadsRevealedSize + header.depthOffsetsRevealedSize));
        }

        // Read updated quads
        quadsUpdated.resize(header.quadsUpdatedSize);
        std::memcpy(quadsUpdated.data(), ptr, header.quadsUpdatedSize);
        ptr += header.quadsUpdatedSize;

        // Read updated depth offsets
        depthOffsetsUpdated.resize(header.depthOffsetsUpdatedSize);
        std::memcpy(depthOffsetsUpdated.data(), ptr, header.depthOffsetsUpdatedSize);
        ptr += header.depthOffsetsUpdatedSize;

        // Read revealed quads
        quadsRevealed.resize(header.quadsRevealedSize);
        std::memcpy(quadsRevealed.data(), ptr, header.quadsRevealedSize);
        ptr += header.quadsRevealedSize;

        // Read revealed depth offsets
        depthOffsetsRevealed.resize(header.depthOffsetsRevealedSize);
        std::memcpy(depthOffsetsRevealed.data(), ptr, header.depthOffsetsRevealedSize);
        ptr += header.depthOffsetsRevealedSize;

        return quadsUpdated.size() + depthOffsetsUpdated.size() +
               quadsRevealed.size() + depthOffsetsRevealed.size();
    }

private:
    ZSTDCodec resQuadsUpdatedCodec;
    ZSTDCodec resOffsetsUpdatedCodec;
    ZSTDCodec resQuadsRevealedCodec;
    ZSTDCodec resOffsetsRevealedCodec;
};

} // namespace quasar

#endif // QUAD_FRAME_H
