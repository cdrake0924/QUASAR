#ifndef NETWORK_UTILS_H
#define NETWORK_UTILS_H

#include <string>
#include <stdexcept>

namespace quasar {

namespace networkutils {

inline std::pair<std::string, int> parseIPAddressAndPort(const std::string& ipAddressAndPort) {
    size_t pos = ipAddressAndPort.find(':');
    if (pos == std::string::npos) {
        throw std::invalid_argument("Invalid address format, expected \"ip:port\" but got: " + ipAddressAndPort);
    }

    std::string ipAddress = ipAddressAndPort.substr(0, pos);
    int port;
    try {
        port = std::stoi(ipAddressAndPort.substr(pos + 1));
    }
    catch (const std::exception& e) {
        throw std::invalid_argument("Invalid port number: " + ipAddressAndPort);
    }
    return { ipAddress, port };
}

} // namespace networkutils

} // namespace quasar

#endif // NETWORK_UTILS_H
