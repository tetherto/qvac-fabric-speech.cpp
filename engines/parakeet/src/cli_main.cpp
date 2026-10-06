// Standalone program entry; forwards to parakeet_cli_main (see main.cpp).

#include "parakeet/cli.h"

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    try {
        return parakeet_cli_main(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "parakeet: %s\n", error.what());
        return 1;
    }
}
