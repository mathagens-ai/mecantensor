#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mecan/runtime/interfaces.h"

namespace mecan {
namespace runtime {

class DeviceBackendRegistry {
public:
    static DeviceBackendRegistry& instance();

    void register_backend(std::shared_ptr<IDeviceBackend> backend);
    void register_default_backends();

    std::shared_ptr<IDeviceBackend> find_by_name(const std::string& backend_name) const;
    std::shared_ptr<IDeviceBackend> find_best_backend(QuantScheme scheme) const;
    std::vector<std::shared_ptr<IDeviceBackend>> list_available() const;

private:
    DeviceBackendRegistry() = default;

    std::vector<std::shared_ptr<IDeviceBackend>> backends_;
};

} // namespace runtime
} // namespace mecan
