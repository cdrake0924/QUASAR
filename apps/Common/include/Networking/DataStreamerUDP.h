#ifndef DATA_STREAMER_UDP_H
#define DATA_STREAMER_UDP_H

#include <thread>
#include <atomic>
#include <concurrentqueue/concurrentqueue.h>

#include <Utils/TimeUtils.h>
#include <Networking/DataPacketUDP.h>
#include <Networking/Socket.h>

namespace quasar {

class DataStreamerUDP {
public:
    std::string url;

    int maxDataSize;

    DataStreamerUDP(std::string url, int maxDataSize, bool nonBlocking = false);
    ~DataStreamerUDP();

    void close();

    int send(const uint8_t* data);

private:
    SocketUDP socket;

    std::thread dataSendingThread;

    std::atomic_bool running{false};

    packet_id_t dataID = 0;

    moodycamel::ConcurrentQueue<DataPacketUDP> packets;

    int sendPacket(DataPacketUDP* packet);
    void sendData();
};

} // namespace quasar

#endif // DATA_STREAMER_UDP_H
