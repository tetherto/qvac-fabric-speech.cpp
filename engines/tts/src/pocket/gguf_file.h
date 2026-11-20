#pragma once
// A Pocket GGUF opened once for a load (POSIX): the metadata and every
// streamed weight read through this one descriptor, so every byte comes from
// one file whatever the path names meanwhile, and fstat of the descriptor is
// the identity of the file the weights were loaded from. On Windows, when the
// open fails, or for an empty path (a metadata-only measure), every reader
// opens the path as before and no identity is recorded.

#include "gguf.h"
#include "gguf_stream.h"

#include <cstdio>
#include <memory>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tts_cpp::pocket::detail {

struct GgufFile {
    int fd = -1;
    explicit GgufFile(const std::string & path) {
#ifndef _WIN32
        if (!path.empty()) fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    }
    ~GgufFile() {
#ifndef _WIN32
        if (fd >= 0) ::close(fd);
#endif
    }
    GgufFile(const GgufFile &) = delete;
    GgufFile & operator=(const GgufFile &) = delete;

    gguf_context * init(const std::string & path, gguf_init_params gp) const {
#ifndef _WIN32
        if (fd >= 0) {
            const int d = ::dup(fd);
            FILE * f = d >= 0 ? ::fdopen(d, "rb") : nullptr;
            if (!f) {
                if (d >= 0) ::close(d);
                return nullptr;
            }
            gguf_context * g = std::fseek(f, 0, SEEK_SET) == 0 ? gguf_init_from_file_ptr(f, gp) : nullptr;
            std::fclose(f);
            return g;
        }
#endif
        return gguf_init_from_file(path.c_str(), gp);
    }

    std::unique_ptr<::tts_cpp::detail::gguf_stream_reader> stream(const gguf_context * g,
                                                                   const std::string & path) const {
#ifndef _WIN32
        if (fd >= 0) return std::make_unique<::tts_cpp::detail::gguf_stream_reader>(g, fd);
#endif
        return std::make_unique<::tts_cpp::detail::gguf_stream_reader>(g, path);
    }
};

} // namespace tts_cpp::pocket::detail
