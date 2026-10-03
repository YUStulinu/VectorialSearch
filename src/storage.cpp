#include "vecsearch/storage.hpp"

#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#endif

namespace vecsearch {

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept {
    *this = std::move(other);
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this == &other) return *this;
    close();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
#if defined(_WIN32)
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
    other.file_handle_ = nullptr;
    other.map_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    return *this;
}

#if defined(_WIN32)

void MappedFile::open(const std::string& path) {
    close();

    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Could not open \"" + path + "\" (error " + std::to_string(GetLastError()) + ")");
    }

    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(file, &fileSize) || fileSize.QuadPart == 0) {
        CloseHandle(file);
        throw std::runtime_error("\"" + path + "\" is empty or its size could not be read");
    }

    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        const DWORD err = GetLastError();
        CloseHandle(file);
        throw std::runtime_error("CreateFileMapping failed (error " + std::to_string(err) + ")");
    }

    void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        const DWORD err = GetLastError();
        CloseHandle(mapping);
        CloseHandle(file);
        throw std::runtime_error("MapViewOfFile failed (error " + std::to_string(err) + ")");
    }

    file_handle_ = file;
    map_handle_ = mapping;
    data_ = static_cast<const std::uint8_t*>(view);
    size_ = static_cast<std::size_t>(fileSize.QuadPart);
}

void MappedFile::close() {
    if (data_) UnmapViewOfFile(const_cast<std::uint8_t*>(data_));
    if (map_handle_) CloseHandle(map_handle_);
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
    data_ = nullptr;
    size_ = 0;
    map_handle_ = nullptr;
    file_handle_ = nullptr;
}

void MappedFile::adviseRandom() const {
    // Windows chooses access patterns per-handle at open time; there is no
    // madvise equivalent for an existing view, so this is a no-op.
}

#else  // POSIX

void MappedFile::open(const std::string& path) {
    close();

    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Could not open \"" + path + "\": " + std::strerror(errno));
    }

    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        ::close(fd);
        throw std::runtime_error("\"" + path + "\" is empty or could not be stat'ed");
    }

    void* addr = mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
    }

    fd_ = fd;
    data_ = static_cast<const std::uint8_t*>(addr);
    size_ = static_cast<std::size_t>(st.st_size);
}

void MappedFile::close() {
    if (data_) munmap(const_cast<std::uint8_t*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
}

void MappedFile::adviseRandom() const {
    if (data_) madvise(const_cast<std::uint8_t*>(data_), size_, MADV_RANDOM);
}

#endif

}  // namespace vecsearch
