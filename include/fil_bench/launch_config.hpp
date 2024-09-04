#include <ostream>

#include <fil_bench/fwd_decl.hpp>

#include <cuml/experimental/fil/tree_layout.hpp>
#include <cuml/fil/fil.h>
#include <nlohmann/json_fwd.hpp>

#ifndef FIL_BENCH_LAUNCH_CONFIG_HPP_
#define FIL_BENCH_LAUNCH_CONFIG_HPP_

namespace fil_bench {

struct launch_config_t {
  // Configuration common to old and new FIL
  int chunk_size{1};
  // Configuration for old FIL only
  ML::fil::algo_t algo_type{ML::fil::algo_t::NAIVE};
  ML::fil::storage_type_t storage_type{ML::fil::storage_type_t::DENSE};
  // Configuration for new FIL only
  ML::experimental::fil::tree_layout layout{ML::experimental::fil::tree_layout::breadth_first};
};

std::ostream& operator<<(std::ostream& os, launch_config_t const& config);
void to_json(nlohmann::ordered_json& js, launch_config_t const& config);
void from_json(nlohmann::ordered_json const& js, launch_config_t& config);

}  // namespace fil_bench

#endif  // FIL_BENCH_LAUNCH_CONFIG_HPP_
