#include <sstream>
#include <spdlog/spdlog.h>
#include <Streamers/VideoStreamer.h>
#include <Utils/TimeUtils.h>
#include <Networking/Utils.h>

using namespace quasar;

VideoStreamer::VideoStreamer(
        const RenderTargetCreateParams& params,
        const std::string& videoURL,
        int maxFrameRate,
        int targetBitRateMbps)
    : videoURL(videoURL)
    , videoWidth(params.width + poseIDOffset)
    , videoHeight(params.height)
    , maxFrameRate(maxFrameRate)
    , targetBitRateKbps(targetBitRateMbps * 1000)
    , RenderTarget(params)
#if defined(HAS_CUDA)
    , cudaGLImage(colorTexture)
#endif
{
    renderTargetCopy = new RenderTarget({
        .width = width,
        .height = height,
        .internalFormat = colorTexture.internalFormat,
        .format = colorTexture.format,
        .type = colorTexture.type,
        .wrapS = colorTexture.wrapS,
        .wrapT = colorTexture.wrapT,
        .minFilter = colorTexture.minFilter,
        .magFilter = colorTexture.magFilter,
        .multiSampled = colorTexture.multiSampled,
    });

    rgbaVideoFrameData.resize(videoWidth * videoHeight * 4);
#if !defined(HAS_CUDA)
    openglFrameData.resize(width * height * 4);
#endif

    if (videoURL.empty()) {
        return;
    }

    gst_init(nullptr, nullptr);

    GstRegistry *registry = gst_registry_get();
    GList *factories = gst_registry_get_feature_list(registry, GST_TYPE_ELEMENT_FACTORY);
    std::ostringstream codecs;
    for (GList *l = factories; l != nullptr; l = l->next) {
        GstElementFactory *factory = GST_ELEMENT_FACTORY(l->data);
        const gchar *klass = gst_element_factory_get_metadata(factory, GST_ELEMENT_METADATA_KLASS);
        const gchar *name  = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
        if (klass && g_strrstr(klass, "Encoder")) {
            codecs << name << " ";
        }
    }
    spdlog::debug("Available Encoders: {}", codecs.str());
    gst_plugin_feature_list_free(factories);

    auto [host, port] = networkutils::parseIPAddressAndPort(videoURL);

    std::string encoderName;
    const int gopFrames = std::max(1, maxFrameRate); // 1 second GOP at worst case FPS
#if defined(HAS_CUDA)
    encoderName = "nvh264enc preset=4 rc-mode=cbr zerolatency=true "
                  "bframes=0 gop-size=" + std::to_string(gopFrames);
#else
    encoderName = "x264enc speed-preset=veryfast tune=zerolatency byte-stream=true"
                  "bframes=0 key-int-max=" + std::to_string(gopFrames);
#endif

    std::ostringstream oss;
    oss << "appsrc name=" << appSrcName << " is-live=true format=time "
        << "caps=video/x-raw,format=RGBA,width=" << videoWidth
        << ",height=" << videoHeight << " ! "
        << "queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! "
        << "videoconvert ! video/x-raw,format=" << format << " ! "
        << encoderName << " bitrate=" << targetBitRateKbps << " ! "
        << "rtph264pay mtu=1200 config-interval=1 pt=96 name=" << payloaderName << " ! "
        << "udpsink host=" << host << " port=" << port
        << " sync=false async=false";
    std::string pipelineStr = oss.str();

    GError* error = nullptr;
    pipeline = gst_parse_launch(pipelineStr.c_str(), &error);
    if (!pipeline || error) {
        spdlog::error("GStreamer pipeline error: {}", error ? error->message : "unknown");
        g_error_free(error);
        throw std::runtime_error("Failed to create GStreamer pipeline.");
    }

    appsrc = gst_bin_get_by_name(GST_BIN(pipeline), appSrcName.c_str());
    g_object_set(G_OBJECT(appsrc),
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "do-timestamp", TRUE,
                 nullptr);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    videoStreamerThread = std::thread(&VideoStreamer::encodeAndSendFrames, this);
    spdlog::info("Created VideoStreamer (GStreamer) that sends to URL: {}", videoURL);
}

VideoStreamer::~VideoStreamer() {
    if (videoURL.empty()) {
        return;
    }

    stop();
}

void VideoStreamer::stop() {
    shouldTerminate = true;

    if (videoStreamerThread.joinable())
        videoStreamerThread.join();

    if (appsrc) {
        gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
        gst_object_unref(appsrc);
    }

    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }

    delete renderTargetCopy;
}

void VideoStreamer::sendFrame(pose_id_t poseID) {
#if defined(HAS_CUDA)
    bind();
    blit(*renderTargetCopy);
    unbind();

    cudaGLImage.map();
    cudaArray_t cudaBuffer = cudaGLImage.getArrayMapped();
    cudaGLImage.unmap();

    cudaBufferQueue.enqueue({ poseID, cudaBuffer });
#else
    bind();
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, openglFrameData.data());
    unbind();

    cpuBufferQueue.enqueue({ poseID, openglFrameData });
#endif
}

void VideoStreamer::packPoseIDIntoVideoFrame(pose_id_t poseID) {
    for (int i = 0; i < poseIDOffset; i++) {
        uint8_t value = (poseID & (1 << i)) ? 255 : 0;
        for (int j = 0; j < videoHeight; ++j) {
            int index = j * videoWidth * 4 + (videoWidth - 1 - i) * 4;
            rgbaVideoFrameData[index + 0] = value;
            rgbaVideoFrameData[index + 1] = value;
            rgbaVideoFrameData[index + 2] = value;
            rgbaVideoFrameData[index + 3] = 255;
        }
    }
}

void VideoStreamer::encodeAndSendFrames() {
    videoReady = true;

    time_t prevTime = timeutils::getTimeMicros();

    while (!shouldTerminate) {
        double frameIntervalSec = 1.0 / maxFrameRate;
        time_t frameStart = timeutils::getTimeMicros();

#if defined(HAS_CUDA)
        CudaBuffer cudaStruct;
        if (!cudaBufferQueue.try_dequeue(cudaStruct)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        pose_id_t poseIDToSend = cudaStruct.poseID;
        cudaArray_t cudaBuffer = cudaStruct.buffer;

        time_t startTransferTimeMs = timeutils::getTimeMicros();

        CHECK_CUDA_ERROR(cudaMemcpy2DFromArray(rgbaVideoFrameData.data(),
                                               videoWidth * 4,
                                               cudaBuffer,
                                               0, 0,
                                               width * 4, height,
                                               cudaMemcpyDeviceToHost));
#else
        CPUBuffer cpuStruct;
        if (!cpuBufferQueue.try_dequeue(cpuStruct)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        pose_id_t poseIDToSend = cpuStruct.poseID;
        time_t startTransferTimeMs = timeutils::getTimeMicros();

        for (int row = 0; row < height; ++row) {
            std::memcpy(&rgbaVideoFrameData[row * videoWidth * 4],
                        &cpuStruct.data[row * width * 4],
                        width * 4);
        }
#endif

        packPoseIDIntoVideoFrame(poseIDToSend);
        stats.transferTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startTransferTimeMs);

        time_t startEncode = timeutils::getTimeMicros();

        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, rgbaVideoFrameData.size(), nullptr);
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        std::memcpy(map.data, rgbaVideoFrameData.data(), rgbaVideoFrameData.size());
        gst_buffer_unmap(buffer, &map);

        GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
        if (ret != GST_FLOW_OK) {
            spdlog::error("Failed to push buffer to GStreamer: {}", static_cast<int>(ret));
        }
        framesSent++;

        time_t frameEnd = timeutils::getTimeMicros();

        stats.encodeTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startEncode);

        stats.sendTimeMs = timeutils::microsToMillis(frameEnd - frameStart);

        double elapsedTimeSec = timeutils::microsToSeconds(frameEnd - frameStart);
        if (elapsedTimeSec < frameIntervalSec) {
            std::this_thread::sleep_for(std::chrono::microseconds(
                (int)(timeutils::secondsToMicros(frameIntervalSec - elapsedTimeSec))));
        }

        stats.totalSendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - prevTime);
        prevTime = timeutils::getTimeMicros();
    }
}

