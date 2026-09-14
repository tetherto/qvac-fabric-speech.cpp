#include "audio8/codec_bench_cli.h"
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

static bool accepts(std::initializer_list<const char *> args) {
    std::vector<std::string> strings{"audio8-codec-bench"};
    for (const auto * arg : args) strings.emplace_back(arg);
    std::vector<char *> argv;
    for (auto & arg : strings) argv.push_back(&arg[0]);
    try { audio8_codec_bench::parse(static_cast<int>(argv.size()), argv.data()); return true; }
    catch (const std::exception &) { return false; }
}
int main() {
    int failed = 0;
    auto check = [&](bool result, const char * description) {
        if (!result) { std::fprintf(stderr, "FAIL: %s\n", description); ++failed; }
    };
    check(accepts({"--help"}), "help without models");
    check(!accepts({}), "missing required paths");
    check(!accepts({"--bogus"}), "unknown option");
    check(!accepts({"--in"}), "missing option value");
    check(!accepts({"--in", "--out", "x"}), "option consumed as value");
    check(accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--threads", "2", "--n-gpu-layers", "99"}), "valid reconstruction args");
    for (const auto * n : {"0", "-1", "2x", "1.5", "2147483648", "9999999999999999999999"})
        check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--threads", n}), "invalid threads");
    check(accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--n-gpu-layers", "0"}), "explicit CPU");
    for (const auto * n : {"-1", "99x", "2147483648"})
        check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "out.wav", "--n-gpu-layers", n}), "invalid GPU layers");
    check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "in.wav"}), "input overwrite");
    check(!accepts({"--codec-encoder", "enc", "--codec-decoder", "dec", "--in", "in.wav", "--out", "enc"}), "model overwrite");
    if (!failed) std::puts("codec benchmark CLI tests passed");
    return failed ? 1 : 0;
}
