#include <ostream>

#include <fil_bench/launch_config.hpp>

namespace fil_bench {

std::ostream& operator<<(std::ostream& os, launch_config_t const& config) {
  os << "{chunk_size=" << config.chunk_size << ", algo_type=";
  switch (config.algo_type) {
  case ML::fil::algo_t::ALGO_AUTO:
    os << "ML::fil::algo_t::ALGO_AUTO, ";
    break;
  case ML::fil::algo_t::NAIVE:
    os << "ML::fil::algo_t::NAIVE, ";
    break;
  case ML::fil::algo_t::TREE_REORG:
    os << "ML::fil::algo_t::TREE_REORG, ";
    break;
  case ML::fil::algo_t::BATCH_TREE_REORG:
    os << "ML::fil::algo_t::BATCH_TREE_REORG, ";
    break;
  }

  os << "storage_type=";
  switch (config.storage_type) {
  case ML::fil::storage_type_t::AUTO:
    os << "ML::fil::storage_type_t::AUTO, ";
    break;
  case ML::fil::storage_type_t::DENSE:
    os << "ML::fil::storage_type_t::DENSE, ";
    break;
  case ML::fil::storage_type_t::SPARSE:
    os << "ML::fil::storage_type_t::SPARSE, ";
    break;
  case ML::fil::storage_type_t::SPARSE8:
    os << "ML::fil::storage_type_t::SPARSE8, ";
    break;
  }

  os << "layout=";
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
