# Makefile — Linux and macOS. Windows uses CMake or build.bat.
#
# distance_avx2.cpp is the only file compiled with AVX2 enabled. Everything
# else stays baseline x86-64 so the binary still starts on an old CPU and
# picks its kernel at runtime.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Iinclude
AVXFLAGS := -mavx2 -mfma
LDFLAGS  ?= -pthread
OUT      := out

CORE_SRC := src/hnsw.cpp src/storage.cpp src/distance.cpp src/distance_scalar.cpp
CORE_OBJ := $(patsubst src/%.cpp,$(OUT)/%.o,$(CORE_SRC))
AVX_OBJ  := $(OUT)/distance_avx2.o

.PHONY: all test bench clean asan tsan
all: $(OUT)/tests $(OUT)/bench

$(OUT):
	@mkdir -p $(OUT)

$(OUT)/%.o: src/%.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(AVX_OBJ): src/distance_avx2.cpp | $(OUT)
	$(CXX) $(CXXFLAGS) $(AVXFLAGS) -c $< -o $@

$(OUT)/tests: tests/test_hnsw.cpp $(CORE_OBJ) $(AVX_OBJ)
	$(CXX) $(CXXFLAGS) -Itests $^ -o $@ $(LDFLAGS)

$(OUT)/bench: bench/bench.cpp $(CORE_OBJ) $(AVX_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test: $(OUT)/tests
	./$(OUT)/tests

bench: $(OUT)/bench
	./$(OUT)/bench

# Finds buffer overruns and undefined behaviour.
asan: | $(OUT)
	$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude $(AVXFLAGS) -c src/distance_avx2.cpp -o $(OUT)/avx_asan.o
	$(CXX) -std=c++17 -O1 -g -fsanitize=address,undefined -Iinclude -Itests $(CORE_SRC) $(OUT)/avx_asan.o tests/test_hnsw.cpp -o $(OUT)/tests_asan $(LDFLAGS)
	ASAN_OPTIONS=detect_leaks=0 ./$(OUT)/tests_asan

# Finds data races in the multi-threaded build path.
tsan: | $(OUT)
	$(CXX) -std=c++17 -O1 -g -fsanitize=thread -Iinclude $(AVXFLAGS) -c src/distance_avx2.cpp -o $(OUT)/avx_tsan.o
	$(CXX) -std=c++17 -O1 -g -fsanitize=thread -Iinclude -Itests $(CORE_SRC) $(OUT)/avx_tsan.o tests/test_hnsw.cpp -o $(OUT)/tests_tsan $(LDFLAGS)
	./$(OUT)/tests_tsan

clean:
	rm -rf $(OUT)
