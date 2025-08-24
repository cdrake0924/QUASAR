#ifndef DEPTH_STREAMER_H
#define DEPTH_STREAMER_H

#include <thread>
#include <atomic>
#include <vector>
#include <memory>
#include <concurrentqueue/concurrentqueue.h>

#include <RenderTargets/RenderTarget.h>
#include <Networking/DataStreamerTCP.h>
#include <CameraPose.h>

#if defined(HAS_CUDA)
#include <CudaGLInterop/CudaGLImage.h>
#endif

namespace quasar {

class DepthStreamer : public RenderTarget {
public:
    std::string receiverURL;

    int imageSize;

    struct Stats {
        double timeToTransferMs = 0.0;
        double timeToSendMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DepthStreamer(const RenderTargetCreateParams& params, std::string receiverURL);
    ~DepthStreamer();

    void close();

    float getFrameRate() {
        return 1.0f / timeutils::millisToSeconds(stats.timeToSendMs);
    }

    void setTargetFrameRate(int targetFrameRate) {
        this->targetFrameRate = targetFrameRate;
    }

    void sendFrame(pose_id_t poseID);

private:
    int targetFrameRate = 30;
    std::unique_ptr<DataStreamerTCP> streamer;

    std::vector<char> data;
    RenderTarget renderTargetCopy;

#if defined(HAS_CUDA)
    CudaGLImage cudaGLImage;

    struct CudaBuffer {
        pose_id_t poseID;
        cudaArray_t buffer;
    };
    moodycamel::ConcurrentQueue<CudaBuffer> cudaBufferQueue;

    std::thread dataSendingThread;
    std::atomic_bool running{false};

    void sendData();
#else
    pose_id_t poseID;
#endif
};

} // namespace quasar

#endif // DEPTH_STREAMER_H
