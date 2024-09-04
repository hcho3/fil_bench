#ifndef FIL_BENCH_ARRAY_TYPES_HPP_
#define FIL_BENCH_ARRAY_TYPES_HPP_

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>

namespace fil_bench {

using Device1DArray = raft::device_mdarray<float, raft::vector_extent<int>, raft::layout_right>;
using Device1DArrayView = raft::device_mdspan<float, raft::vector_extent<int>, raft::layout_right>;
using Device2DArray = raft::device_mdarray<float, raft::matrix_extent<int>, raft::layout_right>;
using Device2DArrayView = raft::device_mdspan<float, raft::matrix_extent<int>, raft::layout_right>;

}  // namespace fil_bench

#endif  // FIL_BENCH_ARRAY_TYPES_HPP_
