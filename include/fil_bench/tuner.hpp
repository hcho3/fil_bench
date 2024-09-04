#ifndef FIL_BENCH_TUNER_HPP_
#define FIL_BENCH_TUNER_HPP_

#include <fil_bench/array_types.hpp>
#include <fil_bench/fwd_decl.hpp>
#include <fil_bench/launch_config.hpp>

namespace fil_bench {

launch_config_t optimize_old_fil(
    raft::handle_t& handle, treelite::Model* tl_model, Device2DArrayView X);

launch_config_t optimize_new_fil(
    raft::handle_t& handle, treelite::Model* tl_model, Device2DArrayView X);

}  // namespace fil_bench

#endif  // FIL_BENCH_TUNER_HPP_
