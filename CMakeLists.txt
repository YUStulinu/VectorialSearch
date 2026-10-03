# CMakeLists.txt — the portable build. Works with Visual Studio, GCC and Clang.
#
#   cmake -B cmake-build
#   cmake --build cmake-build --config Release
#   ./cmake-build/tests        (Windows: cmake-build\Release\tests.exe)
#
# Not "build": node-gyp owns that folder and wipes it on every rebuild.
cmake_minimum_required(VERSION 3.15)
project(vecsearch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(Threads REQUIRED)

# The AVX2 kernels are their own target so the instruction set is enabled for
# exactly one file. Compiling the whole library with AVX2 would produce a
# binary that crashes on any CPU without it, which defeats the runtime
# dispatch entirely.
add_library(vecsearch_avx2 STATIC src/distance_avx2.cpp)
target_include_directories(vecsearch_avx2 PUBLIC include)
if(MSVC)
  target_compile_options(vecsearch_avx2 PRIVATE /arch:AVX2 /O2)
else()
  target_compile_options(vecsearch_avx2 PRIVATE -mavx2 -mfma -O3)
endif()

add_library(vecsearch STATIC
  src/hnsw.cpp
  src/storage.cpp
  src/distance.cpp
  src/distance_scalar.cpp
)
target_include_directories(vecsearch PUBLIC include)
target_link_libraries(vecsearch PUBLIC vecsearch_avx2 Threads::Threads)
if(MSVC)
  target_compile_options(vecsearch PRIVATE /O2 /W4)
  # Needed for std::aligned_alloc-free code paths and sane exception handling.
  target_compile_definitions(vecsearch PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(vecsearch PRIVATE -O3 -Wall -Wextra)
endif()

add_executable(tests tests/test_hnsw.cpp)
target_include_directories(tests PRIVATE tests)
target_link_libraries(tests PRIVATE vecsearch)

add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE vecsearch)

enable_testing()
add_test(NAME correctness COMMAND tests)
