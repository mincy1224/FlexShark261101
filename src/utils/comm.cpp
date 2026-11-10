#include <shark/utils/comm.hpp>
#include <shark/utils/assert.hpp>
#include <iostream>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace shark
{
    namespace
    {
        /// Channel endpoints are Unix domain sockets rather than TCP: WSL2's
        /// mirrored networking resets loopback TCP connections mid-stream, and
        /// going through the socket API keeps the bytes off the network stack
        /// entirely. The path is relative, so both peers must share a working
        /// directory (they do: the key files live there as well).
        std::string channel_path(int port)
        {
            return ".shark-" + std::to_string(port) + ".sock";
        }

        socklen_t unix_addr(struct sockaddr_un &addr, const std::string &path)
        {
            std::memset(&addr, 0, sizeof(addr));
            addr.sun_family = AF_UNIX;
            std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path.c_str());
            return sizeof(addr);
        }

        /// Unix sockets default to the ~200 KiB net.core.w*_default and, unlike
        /// TCP, the kernel never grows them. The protocols move 75 KiB - 384 KiB
        /// blocks per send, so mirror the TCP autotuning ceiling instead.
        void tune_buffer(int fd)
        {
            const int size = 4 << 20;
            setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
            setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
        }

        int connect_channel(int port)
        {
            struct sockaddr_un addr;
            const socklen_t len = unix_addr(addr, channel_path(port));
            while (1)
            {
                const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
                if (fd < 0)
                {
                    perror("socket");
                    exit(1);
                }
                if (connect(fd, (struct sockaddr *)&addr, len) == 0)
                {
                    tune_buffer(fd);
                    return fd;
                }
                ::close(fd);
                usleep(1000);
            }
        }

        int listen_channel(int port)
        {
            const std::string path = channel_path(port);
            ::unlink(path.c_str());
            struct sockaddr_un addr;
            const socklen_t len = unix_addr(addr, path);

            const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
            if (fd < 0)
            {
                perror("socket");
                exit(1);
            }
            if (::bind(fd, (struct sockaddr *)&addr, len) < 0)
            {
                perror("error: bind");
                exit(1);
            }
            if (listen(fd, 8) < 0)
            {
                perror("error: listen");
                exit(1);
            }
            return fd;
        }
    }

    SocketBuf::SocketBuf(std::string ip, int port, bool onlyRecv = false)
    {
        (void)ip; // unused: both channels are Unix domain sockets named by port
        this->t = BUF_SOCKET;
        std::cerr << "trying to connect with server...";
        recvsocket = connect_channel(port);
        sleep(1);
        if (!onlyRecv)
        {
            sendsocket = connect_channel(port + 3);
        }
        std::cerr << "connected" << std::endl;
    }

    void SocketBuf::sync()
    {
        char buf[1] = {1};
        send(sendsocket, buf, 1, 0);
        recv(recvsocket, buf, 1, MSG_WAITALL);
        bytesReceived += 1;
        bytesSent += 1;
        always_assert(buf[0] == 1);
    }

    void SocketBuf::read(char *buf, u64 bytes)
    {
        always_assert(bytes == recv(recvsocket, (char *)buf, bytes, MSG_WAITALL));
        bytesReceived += bytes;

        // u64 received = 0;
        // while (received < bytes) {
        //     int n = recv(recvsocket, buf + received, bytes - received, 0);
        //     if (n == 0) {
        //         fprintf(stderr, "recv: peer closed connection\n");
        //         abort();
        //     }
        //     if (n < 0) {
        //         if (errno == EINTR)
        //             continue;
        //         perror("recv failed");
        //         abort();
        //     }
        //     received += n;
        // }

        // always_assert(bytes == received);
        // bytesReceived += bytes;
    }

    char *SocketBuf::read(u64 bytes)
    {
        char *tmpBuf = new char[bytes];
        always_assert(bytes == recv(recvsocket, (char *)tmpBuf, bytes, MSG_WAITALL));
        bytesReceived += bytes;
        return tmpBuf;

        // char *tmpBuf = new char[bytes];
        // u64 received = 0;
        // while (received < bytes) {
        //     int n = recv(recvsocket, tmpBuf + received, bytes - received, 0);
        //     if (n == 0) {
        //         fprintf(stderr, "recv: peer closed connection\n");
        //         abort();
        //     }
        //     if (n < 0) {
        //         if (errno == EINTR)
        //             continue;
        //         perror("recv failed");
        //         abort();
        //     }
        //     received += n;
        // }

        // always_assert(bytes == received);
        // bytesReceived += bytes;
        // return tmpBuf;
    }

    void SocketBuf::write(char *buf, u64 bytes)
    {
        always_assert(bytes == send(sendsocket, buf, bytes, 0));
        bytesSent += bytes;

        // u64 sent = 0;
        // while (sent < bytes) {
        //     int n = send(sendsocket, buf + sent, bytes - sent, 0);
        //     if (n <= 0) {
        //         perror("send");
        //         abort();
        //     }
        //     sent += n;
        // }
        // always_assert(bytes == sent);
        // bytesSent += bytes;
    }

    void SocketBuf::close()
    {
        ::close(sendsocket);
        ::close(recvsocket);
    }

    void Peer::close()
    {
        keyBuf->close();
    }

    Peer *waitForPeer(int port)
    {
        int sendsocket, recvsocket;
        std::cerr << "waiting for connection from client...";
        {
            const int mysocket = listen_channel(port);
            sendsocket = accept(mysocket, nullptr, nullptr);
            tune_buffer(sendsocket);
            ::unlink(channel_path(port).c_str());
            ::close(mysocket);
        }

        {
            const int mysocket = listen_channel(port + 3);
            recvsocket = accept(mysocket, nullptr, nullptr);
            tune_buffer(recvsocket);
            ::unlink(channel_path(port + 3).c_str());
            ::close(mysocket);
        }

        std::cerr << "connected" << std::endl;
        return new Peer(sendsocket, recvsocket);
    }
}