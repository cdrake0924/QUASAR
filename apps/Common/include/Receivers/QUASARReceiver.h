#ifndef QUASAR_RECEIVER_H
#define QUASAR_RECEIVER_H

#include <BS_thread_pool/BS_thread_pool.hpp>

#include <Path.h>
#include <Quads/QuadSet.h>
#include <Quads/QuadFrames.h>
#include <Quads/QuadMesh.h>
#include <Quads/QuadMaterial.h>

namespace quasar {

class QUASARReceiver {
public:
    struct Stats {
        uint totalTriangles = 0;
        double timeToLoadMs = 0.0;
        double timeToDecompressMs = 0.0;
        double timeToTransferMs = 0.0;
        double timeToCreateMeshMs = 0.0;
        QuadSet::Sizes sizes{};
    } stats;

    uint maxLayers;
    std::vector<ReferenceFrame> frames;

    QUASARReceiver(QuadSet& quadSet, uint maxLayers, float remoteFOV, float remoteFOVWide);
    ~QUASARReceiver() = default;

    QuadMesh& getMesh(int layer);

    PerspectiveCamera& getRemoteCamera();
    PerspectiveCamera& getRemoteCameraWideFov();

    void loadFromFiles(const Path& dataPath);

private:
    QuadSet& quadSet;

    PerspectiveCamera remoteCamera;
    PerspectiveCamera remoteCameraWideFov;

    QuadMaterial quadMaterial;
    std::vector<Texture> colorTextures;
    std::vector<QuadMesh> meshes;

    std::unique_ptr<BS::thread_pool<>> threadPool;

    // Temporary buffers for decompression
    std::vector<char> uncompressedQuads, uncompressedOffsets;

    inline const PerspectiveCamera& getCameraToUse(int layer) const {
        return (layer == maxLayers - 1) ? remoteCameraWideFov : remoteCamera;
    }
};

} // namespace quasar

#endif // QUASAR_RECEIVER_H
