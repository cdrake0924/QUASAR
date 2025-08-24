#include <Networking/DataStreamerUDP.h>

using namespace quasar;

DataStreamerUDP::DataStreamerUDP(std::string url, int maxDataSize, bool nonBlocking)
    : url(url)
    , maxDataSize(maxDataSize)
    , socket(nonBlocking)
{
    socket.setAddress(url);

    running = true;
    dataSendingThread = std::thread(&DataStreamerUDP::sendData, this);
}

DataStreamerUDP::~DataStreamerUDP() {
    close();
}

void DataStreamerUDP::close() {
    running = false;

    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
}

int DataStreamerUDP::send(const uint8_t* data) {
    int packetID = 0;
    for (int i = 0; i < maxDataSize; i += PACKET_DATA_SIZE_UDP) {
        DataPacketUDP packet{};
        packet.ID = packetID++;
        packet.dataID = dataID;
        packet.size = std::min(PACKET_DATA_SIZE_UDP, maxDataSize - i);
        std::memcpy(packet.data, data + i, packet.size);

        packets.enqueue(packet);
    }

    dataID++;

    return maxDataSize;
}

int DataStreamerUDP::sendPacket(DataPacketUDP* packet) {
    return socket.send(packet, sizeof(DataPacketUDP), 0);
}

void DataStreamerUDP::sendData() {
    while (running) {
        DataPacketUDP packet;
        if (!packets.try_dequeue(packet)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        int sent = sendPacket(&packet);
        if (sent < 0) {
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                continue;
            }
        }
    }

    socket.close();
}
