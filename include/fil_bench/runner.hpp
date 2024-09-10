#ifndef FIL_BENCH_RUNNER_HPP_
#define FIL_BENCH_RUNNER_HPP_

#include <cstdint>
#include <utility>

#include <fil_bench/array_types.hpp>
#include <fil_bench/fwd_decl.hpp>
#include <fil_bench/launch_config.hpp>

namespace fil_bench {

std::int64_t run_old_fil(raft::handle_t& handle, launch_config_t launch_config,
    treelite::Model* rf_model, Device2DArrayView X, std::uint32_t n_reps);

std::int64_t run_new_fil(raft::handle_t& handle, launch_config_t launch_config,
    treelite::Model* rf_model, Device2DArrayView X, std::uint32_t n_reps);

}  // namespace fil_bench

#endif  // FIL_BENCH_RUNNER_HPP_
