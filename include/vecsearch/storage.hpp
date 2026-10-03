// storage.hpp — read-only memory mapping, on Windows and POSIX.
//
// Why map rather than read:
//
//  * Opening is instant regardless of size. A 6 GB index becomes usable in
//    microseconds, because nothing is read until it is touched.
//  * Pages arrive on demand. A search visits a few thousand vectors, so only
//    those pages are faulted in — not the whole file.
//  * The page cache is shared. Ten processes opening the same index hold one
//    copy in RAM between them, not ten.
//  * The OS can drop clean pages under memory pressure instead of swapping,
//    because it knows it can read them back from the file.
//
// The cost is that the data is read-only and the file must stay put while it
// is mapped.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vecsearch {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    /// Maps the whole file read-only. Throws std::runtime_error on failure.
    void open(const std::string& path);
    void close();

    bool isOpen() const noexcept { return data_ != nullptr; }
    const std::uint8_t* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    /// Hints that the mapping will be read in no particular order, so the
    /// kernel should not bother with sequential read-ahead. A graph walk is
    /// the definition of random access.
    void adviseRandom() const;

private:
    const std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    void* file_handle_ = nullptr;
    void* map_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

/// The header written at the start of an index file.
///
/// Fixed-width types and explicit padding, so a file written by one build is
/// readable by another. Everything after the header is a plain array, placed
/// at an offset the header records.
#pragma pack(push, 1)
struct FileHeader {
    char magic[8];              // "VECSRCH1"
    std::uint32_t version;
    std::uint32_t metric;
    std::uint64_t dim;
    std::uint64_t padded_dim;
    std::uint64_t count;
    std::uint64_t M;
    std::uint64_t M0;
    std::uint64_t ef_construction;
    std::uint64_t entry_point;
    std::int64_t max_level;
    std::uint64_t deleted_count;

    // Byte offsets from the start of the file.
    std::uint64_t vectors_offset;
    std::uint64_t levels_offset;
    std::uint64_t labels_offset;
    std::uint64_t deleted_offset;
    std::uint64_t links0_offset;
    std::uint64_t upper_offset_offset;
    std::uint64_t upper_blob_offset;
    std::uint64_t upper_blob_count;
    std::uint64_t file_size;
};
#pragma pack(pop)

inline constexpr char kMagic[8] = {'V', 'E', 'C', 'S', 'R', 'C', 'H', '1'};
inline constexpr std::uint32_t kFormatVersion = 1;

/// Vectors start here so that every row begins on a 64-byte boundary.
inline constexpr std::uint64_t kVectorAlignment = 64;

inline std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

}  // namespace vecsearch
