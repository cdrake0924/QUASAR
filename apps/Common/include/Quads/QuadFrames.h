#ifndef QUAD_FRAME_H
#define QUAD_FRAME_H

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
    size_t numQuads;
    size_t numDepthOffsets;
    std::vector<char> quads;
    std::vector<char> depthOffsets;

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
};

} // namespace quasar

#endif // QUAD_FRAME_H
