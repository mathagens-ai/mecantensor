#include "io/serialization.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace mecan {
namespace io {

namespace {
constexpr const char* kMtMagic = "MECAN_MT1";
constexpr int32_t kMtVersion = 1;

template <typename T>
void write_pod(std::ofstream& fs, const T& value) {
    fs.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_pod(std::ifstream& fs, T& value) {
    fs.read(reinterpret_cast<char*>(&value), sizeof(T));
}
} // namespace

    void save(const Tensor& tensor, const std::string& filepath) {
        std::ofstream fs(filepath, std::ios::binary);
        if (!fs) throw std::runtime_error("MT Error: Could not open file for saving: " + filepath);

        // Header
        fs.write(kMtMagic, 9);

        write_pod(fs, kMtVersion);

        int32_t dtype = (int32_t)tensor.dtype();
        write_pod(fs, dtype);

        int32_t ndim = (int32_t)tensor.ndimension();
        write_pod(fs, ndim);

        for (size_t s : tensor.shape()) {
            int64_t dim = (int64_t)s;
            write_pod(fs, dim);
        }

        // Body
        size_t nbytes = tensor.numel() * core::element_size(tensor.dtype());
        // For simplicity, save CPU data. 
        fs.write((const char*)tensor.data_ptr<char>(), nbytes);
        
        fs.close();
    }

    Tensor load(const std::string& filepath) {
        std::ifstream fs(filepath, std::ios::binary);
        if (!fs) throw std::runtime_error("MT Error: Could not open file for loading: " + filepath);

        char magic[10] = {0};
        fs.read(magic, 9);
        if (std::strncmp(magic, kMtMagic, 9) != 0) {
            throw std::runtime_error("MT Error: Invalid file magic.");
        }

        int32_t version, dtype_raw, ndim;
        read_pod(fs, version);
        if (version != kMtVersion) {
            throw std::runtime_error("MT Error: Unsupported .mt version.");
        }
        read_pod(fs, dtype_raw);
        read_pod(fs, ndim);

        std::vector<size_t> shape;
        for (int i = 0; i < ndim; ++i) {
            int64_t dim;
            read_pod(fs, dim);
            shape.push_back((size_t)dim);
        }

        core::ScalarType dtype = (core::ScalarType)dtype_raw;
        Tensor T(shape, dtype);

        size_t nbytes = T.numel() * core::element_size(dtype);
        fs.read((char*)T.data_ptr<char>(), nbytes);

        fs.close();
        return T;
    }

    void save_paged_mt(
        const Tensor& tensor,
        runtime::PagedTensorMetadata metadata,
        const std::string& filepath) {
        std::ofstream fs(filepath, std::ios::binary);
        if (!fs) throw std::runtime_error("MT Error: Could not open file for saving: " + filepath);

        if (metadata.tensor_name.empty()) {
            metadata.tensor_name = "tensor";
        }
        if (metadata.shape.empty()) {
            metadata.shape = tensor.shape();
        }
        metadata.total_nbytes = tensor.numel() * core::element_size(tensor.dtype());
        metadata.storage_dtype = tensor.dtype();

        fs.write(kMtMagic, 8);
        write_pod(fs, kMtVersion);

        const uint32_t name_len = static_cast<uint32_t>(metadata.tensor_name.size());
        write_pod(fs, name_len);
        fs.write(metadata.tensor_name.data(), name_len);

        const int32_t dtype = static_cast<int32_t>(metadata.storage_dtype);
        const int32_t preferred_device = static_cast<int32_t>(metadata.preferred_device);
        const int32_t quant_scheme = static_cast<int32_t>(metadata.quant_scheme);
        const uint32_t ndim = static_cast<uint32_t>(metadata.shape.size());

        write_pod(fs, dtype);
        write_pod(fs, preferred_device);
        write_pod(fs, quant_scheme);
        write_pod(fs, ndim);
        write_pod(fs, metadata.page_layout.page_bytes);
        write_pod(fs, metadata.page_layout.alignment_bytes);
        write_pod(fs, metadata.page_layout.compressed);
        write_pod(fs, metadata.total_nbytes);

        for (size_t dim : metadata.shape) {
            uint64_t d = static_cast<uint64_t>(dim);
            write_pod(fs, d);
        }

        const uint64_t page_sz = metadata.page_layout.page_bytes;
        const uint64_t page_count = (metadata.total_nbytes + page_sz - 1) / page_sz;
        write_pod(fs, page_count);

        const char* raw = tensor.data_ptr<char>();
        for (uint64_t page = 0; page < page_count; ++page) {
            const uint64_t offset = page * page_sz;
            const uint64_t bytes = std::min(page_sz, metadata.total_nbytes - offset);
            write_pod(fs, bytes);
            fs.write(raw + offset, static_cast<std::streamsize>(bytes));
        }
    }

    Tensor load_paged_mt(
        const std::string& filepath,
        runtime::PagedTensorMetadata* out_metadata) {
        std::ifstream fs(filepath, std::ios::binary);
        if (!fs) throw std::runtime_error("MT Error: Could not open file for loading: " + filepath);

        char magic[9] = {0};
        fs.read(magic, 8);
        if (std::strncmp(magic, kMtMagic, 8) != 0) {
            throw std::runtime_error("MT Error: Invalid file magic.");
        }

        int32_t version = 0;
        read_pod(fs, version);
        if (version != kMtVersion) {
            throw std::runtime_error("MT Error: Unsupported .mt version.");
        }

        runtime::PagedTensorMetadata meta{};
        uint32_t name_len = 0;
        read_pod(fs, name_len);
        meta.tensor_name.resize(name_len);
        if (name_len > 0) {
            fs.read(&meta.tensor_name[0], name_len);
        }

        int32_t dtype = 0;
        int32_t preferred_device = 0;
        int32_t quant_scheme = 0;
        uint32_t ndim = 0;
        read_pod(fs, dtype);
        read_pod(fs, preferred_device);
        read_pod(fs, quant_scheme);
        read_pod(fs, ndim);
        read_pod(fs, meta.page_layout.page_bytes);
        read_pod(fs, meta.page_layout.alignment_bytes);
        read_pod(fs, meta.page_layout.compressed);
        read_pod(fs, meta.total_nbytes);

        meta.storage_dtype = static_cast<core::ScalarType>(dtype);
        meta.preferred_device = static_cast<core::DeviceType>(preferred_device);
        meta.quant_scheme = static_cast<runtime::QuantScheme>(quant_scheme);

        meta.shape.resize(ndim);
        for (uint32_t i = 0; i < ndim; ++i) {
            uint64_t d = 0;
            read_pod(fs, d);
            meta.shape[i] = static_cast<size_t>(d);
        }

        uint64_t page_count = 0;
        read_pod(fs, page_count);

        Tensor t(meta.shape, meta.storage_dtype);
        char* raw = t.data_ptr<char>();
        for (uint64_t page = 0; page < page_count; ++page) {
            uint64_t bytes = 0;
            read_pod(fs, bytes);
            const uint64_t offset = page * meta.page_layout.page_bytes;
            fs.read(raw + offset, static_cast<std::streamsize>(bytes));
        }

        if (out_metadata) {
            *out_metadata = meta;
        }
        return t;
    }

} // namespace io
} // namespace mecan
