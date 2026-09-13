// glm_tests entry point. Runs the functional self-test suite.
// Fixture directory is read from GLM_TEST_FIXTURES (set by CTest) or defaults
// to tests/fixtures relative to the project root.

#include "self_test.h"
#include "version.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::printf("HummingFlight %s - glm_tests\n", GLM_VERSION_STRING);
    std::string fixtureDir = "tests/fixtures";
    const char* env = std::getenv("GLM_TEST_FIXTURES");
    if (env && *env) fixtureDir = env;

    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--verbose" || a == "-v") verbose = true;
    }

    int failures = glm::runSelfTests(verbose, fixtureDir);
    std::printf("EXIT_STATUS=%s\n", failures == 0 ? "PASS" : "FAIL");

    // CTest convention: nonzero exit on failure.
    return failures == 0 ? 0 : 1;
}