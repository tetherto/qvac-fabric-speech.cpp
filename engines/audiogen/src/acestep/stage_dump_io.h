#pragma once

// Reader/writer for the --dump-stages tensor format shared by the smoke and
// parity harnesses: int32 hdr[3] = {ndim, d0, d1}, then d0 * d1 float32.

#include <cstdint>
#include <cstdio>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr int32_t STAGE_DUMP_RANK = 2;

inline bool read_stage_dump(const char * tag, const char * path, std::vector<float> & out, int * d0, int * d1) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[%s] cannot open %s\n", tag, path);
        return false;
    }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, f) != 3 || hdr[0] != STAGE_DUMP_RANK || hdr[1] <= 0 || hdr[2] <= 0) {
        fprintf(stderr, "[%s] %s is not a rank-2 stage dump\n", tag, path);
        fclose(f);
        return false;
    }
    const size_t n = (size_t) hdr[1] * hdr[2];
    out.resize(n);
    const bool ok = fread(out.data(), sizeof(float), n, f) == n;
    fclose(f);
    if (!ok) {
        fprintf(stderr, "[%s] %s is truncated\n", tag, path);
        return false;
    }
    *d0 = hdr[1];
    *d1 = hdr[2];
    return true;
}

inline bool write_stage_dump(const char * tag, const char * path, const std::vector<float> & v, int d0, int d1) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[%s] cannot write %s\n", tag, path);
        return false;
    }
    const int32_t hdr[3] = { STAGE_DUMP_RANK, d0, d1 };
    fwrite(hdr, sizeof(int32_t), 3, f);
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
    return true;
}

}  // namespace tts_cpp::acestep
