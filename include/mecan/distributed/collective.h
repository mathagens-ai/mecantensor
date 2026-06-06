#pragma once

#include <memory>
#include <string>

#include "mecan/runtime/interfaces.h"

namespace mecan {
namespace distributed {

enum class CollectiveBackend {
    Auto,
    MPI,
    NCCL,
    RCCL,
    Gloo,
    Local
};

std::shared_ptr<runtime::ICollectiveTransport> create_collective_transport(CollectiveBackend backend);

} // namespace distributed
} // namespace mecan
