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

class ReferenceFrame {
public:
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

    size_t saveToFiles(const Path& outputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Save quads
        double startTime = timeutils::getTimeMicros();
        Path quadsFileName = (outputPath / ("quads" + idxStr)).withExtension(".bin.zstd");
        FileIO::saveToBinaryFile(quadsFileName, quads.data(), quads.size());
        spdlog::info("Saved {} quads ({:.3f}MB) in {:.3f}ms",
                       numQuads, static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save depth offsets
        startTime = timeutils::getTimeMicros();
        Path depthOffsetsFileName = (outputPath / ("depthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::saveToBinaryFile(depthOffsetsFileName, depthOffsets.data(), depthOffsets.size());
        spdlog::info("Saved {} depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsets, static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }

    size_t loadFromFiles(const Path& inputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Load quads
        double startTime = timeutils::getTimeMicros();
        Path quadsFileName = (inputPath / ("quads" + idxStr)).withExtension(".bin.zstd");
        auto quadsData = FileIO::loadBinaryFile(quadsFileName);
        quads = std::move(quadsData);
        spdlog::info("Loaded quads ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(quads.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        startTime = timeutils::getTimeMicros();
        Path depthOffsetsFileName = (inputPath / ("depthOffsets" + idxStr)).withExtension(".bin.zstd");
        auto depthOffsetsData = FileIO::loadBinaryFile(depthOffsetsFileName);
        depthOffsets = std::move(depthOffsetsData);
        spdlog::info("Loaded depth offsets ({:.3f}MB) in {:.3f}ms",
                       static_cast<double>(depthOffsets.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        return quads.size() + depthOffsets.size();
    }

private:
    ZSTDCodec refQuadsCodec;
    ZSTDCodec refOffsetsCodec;
};

class ResidualFrame {
public:
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

    size_t saveToFiles(const Path& outputPath, int index = -1) {
        std::string idxStr = (index != -1 ? std::to_string(index) : "");

        // Save updated quads
        double startTime = timeutils::getTimeMicros();
        Path updatedQuadsFileName = (outputPath / ("updatedQuads" + idxStr)).withExtension(".bin.zstd");
        FileIO::saveToBinaryFile(updatedQuadsFileName, quadsUpdated.data(), quadsUpdated.size());
        spdlog::info("Saved {} updated quads ({:.3f}MB) in {:.3f}ms",
                       numQuadsUpdated, static_cast<double>(quadsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save updated depth offsets
        startTime = timeutils::getTimeMicros();
        Path updatedDepthOffsetsFileName = (outputPath / ("updatedDepthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::saveToBinaryFile(updatedDepthOffsetsFileName, depthOffsetsUpdated.data(), depthOffsetsUpdated.size());
        spdlog::info("Saved {} updated depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsUpdated, static_cast<double>(depthOffsetsUpdated.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed quads
        startTime = timeutils::getTimeMicros();
        Path revealedQuadsFileName = (outputPath / ("revealedQuads" + idxStr)).withExtension(".bin.zstd");
        FileIO::saveToBinaryFile(revealedQuadsFileName, quadsRevealed.data(), quadsRevealed.size());
        spdlog::info("Saved {} revealed quads ({:.3f}MB) in {:.3f}ms",
                        numQuadsRevealed, static_cast<double>(quadsRevealed.size()) / BYTES_PER_MEGABYTE,
                          timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

        // Save revealed depth offsets
        startTime = timeutils::getTimeMicros();
        Path revealedDepthOffsetsFileName = (outputPath / ("revealedDepthOffsets" + idxStr)).withExtension(".bin.zstd");
        FileIO::saveToBinaryFile(revealedDepthOffsetsFileName, depthOffsetsRevealed.data(), depthOffsetsRevealed.size());
        spdlog::info("Saved {} revealed depth offsets ({:.3f}MB) in {:.3f}ms",
                       numDepthOffsetsRevealed, static_cast<double>(depthOffsetsRevealed.size()) / BYTES_PER_MEGABYTE,
                         timeutils::microsToMillis(timeutils::getTimeMicros() - startTime));

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
