#ifndef FIL_BENCH_TUNER_HPP_
#define FIL_BENCH_TUNER_HPP_

#include <cstdint>

#include <fil_bench/launch_config.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>

namespace treelite {

class Model;

}  // namespace treelite

namespace fil_bench {

using Device2DArray
    = raft::device_mdarray<float, raft::matrix_extent<std::uint64_t>, raft::layout_right>;
using Device2DArrayView
    = raft::device_mdspan<float, raft::matrix_extent<std::uint64_t>, raft::layout_right>;

launch_config_t optimize_old_fil(
    raft::handle_t& handle, treelite::Model* tl_model, Device2DArrayView X);

launch_config_t optimize_new_fil(
    raft::handle_t& handle, treelite::Model* tl_model, Device2DArrayView X);

}  // namespace fil_bench

#endif  // FIL_BENCH_TUNER_HPP_
