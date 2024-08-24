#include <ostream>

#include <fil_bench/launch_config.hpp>

namespace fil_bench {

std::ostream& operator<<(std::ostream& os, launch_config_t const& config) {
  os << "{chunk_size=" << config.chunk_size << ", algo_type=" << config.algo_type
     << ", storage_type=" << config.storage_type << ", layout=";
  switch (config.layout) {
  case ML::experimental::fil::tree_layout::depth_first:
    os << "ML::experimental::fil::tree_layout::depth_first}";
    break;
  case ML::experimental::fil::tree_layout::breadth_first:
    os << "ML::experimental::fil::tree_layout::breadth_first}";
    break;
  }

  return os;
}

}  // namespace fil_bench
