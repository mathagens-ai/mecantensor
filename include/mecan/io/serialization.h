#pragma once

#include "tensor.h"
#include "mecan/runtime/descriptors.h"
#include <fstream>
#include <string>

namespace mecan {
namespace io {

    /**
     * MT Binary Serialization Format (.mt)
     * Header: [MagicString][Version][DType][NDim][Shape...]
     * Body:   [RawDataPointer]
     */

    void save(const Tensor& tensor, const std::string& filepath);
    Tensor load(const std::string& filepath);

    // Paged MECAN format (.mt):
    // native weight binary codec for MecanTensor (analogous role to .pt in PyTorch),
    // with metadata + fixed-size pages for large-scale streamed models.
    void save_paged_mt(
        const Tensor& tensor,
        runtime::PagedTensorMetadata metadata,
        const std::string& filepath);

    Tensor load_paged_mt(
        const std::string& filepath,
        runtime::PagedTensorMetadata* out_metadata = nullptr);

} // namespace io
} // namespace mecan
