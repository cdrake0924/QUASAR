#ifndef DATA_RECEIVER_TCP_H
#define DATA_RECEIVER_TCP_H

#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <string>
#include <functional>

#include <Networking/Socket.h>

namespace quasar {

class DataReceiverTCP {
public:
    struct Stats {
        double timeToReceiveMs = 0.0;
        double bitrateMbps = 0.0;
    };

    DataReceiverTCP(const std::string& url, bool nonBlocking = false);
    virtual ~DataReceiverTCP();

    void start();
    void close();

protected:
    std::string url;
    Stats stats;
    std::atomic_bool ready = false;

    virtual void onDataReceived(const std::vector<char>& data) = 0;

private:
    SocketTCP socket;
    std::thread dataRecvingThread;

    void recvData();
};

} // namespace quasar

#endif // DATA_RECEIVER_TCP_H
