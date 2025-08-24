#ifndef DATA_STREAMER_TCP_H
#define DATA_STREAMER_TCP_H

#include <thread>
#include <atomic>
#include <concurrentqueue/concurrentqueue.h>

#include <Utils/TimeUtils.h>
#include <Networking/DataPacketUDP.h>
#include <Networking/Socket.h>

namespace quasar {

class DataStreamerTCP {
public:
    std::string url;

    int maxDataSize;

    struct Stats {
        double timeToSendMs = 0.0;
        double bitrateMbps = 0.0;
    } stats;

    DataStreamerTCP(std::string url, int maxDataSize = 65535, bool nonBlocking = false);
    ~DataStreamerTCP();

    int send(std::vector<char>& data, bool copy = false);

private:
    SocketTCP socket;

    std::thread dataSendingThread;

    std::atomic_bool ready = false;
    std::atomic_bool shouldTerminate = false;

    moodycamel::ConcurrentQueue<std::vector<char>> datas;

    void sendData();

    int clientSocketID = -1;
};

} // namespace quasar

#endif // DATA_STREAMER_TCP_H
