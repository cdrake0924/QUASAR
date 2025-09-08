#include <Networking/DataStreamerTCP.h>

using namespace quasar;

DataStreamerTCP::DataStreamerTCP(std::string url, int maxDataSize, bool nonBlocking)
    : url(url)
    , maxDataSize(maxDataSize)
{
    if (url.empty()) {
        return;
    }

    socket = std::make_unique<SocketTCP>(nonBlocking);
    socket->setReuseAddr();
    socket->setSendSize(maxDataSize);

    dataSendingThread = std::thread(&DataStreamerTCP::sendData, this);
}

DataStreamerTCP::~DataStreamerTCP() {
    stop();
}

void DataStreamerTCP::stop() {
    shouldTerminate = true;
    ready = false;
    if (dataSendingThread.joinable()) {
        dataSendingThread.join();
    }
}

int DataStreamerTCP::send(std::vector<char>& data, bool copy) {
    if (!ready) {
        return -1;
    }

    if (copy) {
        datas.enqueue(data);
    }
    else {
        datas.enqueue(std::move(data));
    }

    return data.size();
}

void DataStreamerTCP::sendData() {
    socket->bind(url);
    socket->listen(1);

    while (!shouldTerminate) {
        if ((clientSocketID = socket->accept()) < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else {
            ready = true;
            break;
        }
    }

    while (ready && !shouldTerminate) {
        std::vector<char> data;
        if (!datas.try_dequeue(data)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        int startSendTime = timeutils::getTimeMicros();

        int dataSize = data.size();
        std::vector<char> header(sizeof(dataSize));
        std::memcpy(header.data(), &dataSize, sizeof(dataSize));

        int totalSent = 0;
        while (totalSent < header.size()) {
            int sent = socket->sendToClient(clientSocketID, header.data() + totalSent, header.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) continue;
            }
            else {
                totalSent += sent;
            }
        }

        totalSent = 0;
        while (totalSent < data.size()) {
            int sent = socket->sendToClient(clientSocketID, data.data() + totalSent, data.size() - totalSent, 0);
            if (sent < 0) {
                if (errno == EWOULDBLOCK || errno == EAGAIN) continue;
            }
            else {
                totalSent += sent;
            }
        }

        stats.sendTimeMs = timeutils::microsToMillis(timeutils::getTimeMicros() - startSendTime);
        stats.bitrateMbps = ((sizeof(dataSize) + data.size() * 8.0) / timeutils::millisToSeconds(stats.sendTimeMs)) / BYTES_PER_MEGABYTE;
    }

    socket->close();
}
