// A test harness small enough to read in one sitting, so the project has no
// test-framework dependency to install or build.
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace harness {

struct Case { std::string name; std::function<void()> fn; };
inline std::vector<Case>& cases() { static std::vector<Case> c; return c; }

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { cases().push_back({name, std::move(fn)}); }
};

struct Failure : std::runtime_error { using std::runtime_error::runtime_error; };

inline void check(bool ok, const std::string& what) {
    if (!ok) throw Failure(what);
}

template <typename A, typename B>
void equal(const A& a, const B& b, const std::string& what) {
    if (!(a == b)) throw Failure(what + " (got " + std::to_string(a) + ", expected " + std::to_string(b) + ")");
}

inline void near(double a, double b, double tol, const std::string& what) {
    if (std::fabs(a - b) > tol) {
        throw Failure(what + " (got " + std::to_string(a) + ", expected " + std::to_string(b) +
                      " +/- " + std::to_string(tol) + ")");
    }
}

inline void atLeast(double value, double floor, const std::string& what) {
    if (!(value >= floor)) {
        throw Failure(what + " (got " + std::to_string(value) + ", needed at least " + std::to_string(floor) + ")");
    }
}

inline int run() {
    int failed = 0;
    for (auto& c : cases()) {
        try {
            c.fn();
            std::printf("  \033[32mok\033[0m   %s\n", c.name.c_str());
        } catch (const std::exception& e) {
            std::printf("  \033[31mFAIL\033[0m %s\n       %s\n", c.name.c_str(), e.what());
            ++failed;
        }
    }
    std::printf("\n%zu tests, %d failed\n", cases().size(), failed);
    return failed ? 1 : 0;
}

}  // namespace harness

#define TEST(name)                                                       \
    static void name();                                                  \
    static harness::Registrar reg_##name(#name, name);                   \
    static void name()
