#ifndef FILE_IO_H
#define FILE_IO_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#ifdef __ANDROID__
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/native_activity.h>
#endif

namespace quasar {

class FileIO {
public:
    static void flipVerticallyOnLoad(bool flip);
    static void flipVerticallyOnWrite(bool flip);

    static std::ifstream::pos_type getFileSize(const std::string& filename);

    static std::string loadTextFile(const std::string& filename, uint* sizePtr = nullptr);
    static std::vector<char> loadBinaryFile(const std::string& filename, uint* sizePtr = nullptr);
    static unsigned char* loadImage(const std::string& filename, int* width, int* height, int* channels, int desiredChannels = 0);
    static unsigned char* loadImageFromMemory(const unsigned char* data, int size, int* width, int* height, int* channels, int desiredChannels = 0);
    static float* loadImageHDR(const std::string& filename, int* width, int* height, int* channels, int desiredChannels = 0);

    static size_t saveToTextFile(const std::string& filename, const std::string& data, bool append = false);
    static size_t saveToBinaryFile(const std::string& filename, const void* data, size_t size, bool append = false);
    static void saveToPNG(const std::string& filename, int width, int height, int channels, const void *data);
    static void saveToJPG(const std::string& filename, int width, int height, int channels, const void *data, int quality = 90);
    static void saveToHDR(const std::string& filename, int width, int height, int channels, const float *data);

    static void freeImage(void* imageData);

#ifdef __ANDROID__
    static void registerIOSystem(ANativeActivity* activity);
    static ANativeActivity* getNativeActivity() {
        return activity;
    }
    static AAssetManager* getAssetManager() {
        return activity->assetManager;
    }

    static std::string copyFileToCache(std::string filename);
#endif

private:
#ifdef __ANDROID__
    static ANativeActivity* activity;
#endif
};

} // namespace quasar

#endif // FILE_IO_H
