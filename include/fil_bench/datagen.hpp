#ifndef FIL_BENCH_DATAGEN_HPP_
#define FIL_BENCH_DATAGEN_HPP_

#include <cstdint>
#include <memory>
#include <utility>

#include <fil_bench/array_types.hpp>
#include <fil_bench/constants.hpp>
#include <fil_bench/fwd_decl.hpp>

namespace fil_bench {

std::pair<Device2DArray, Device1DArray> make_regression(
    raft::handle_t const& handle, std::uint64_t n_rows, std::uint64_t n_cols);
std::pair<Device2DArray, Device1DArray> make_empty(
    raft::handle_t const& handle, std::uint64_t n_rows, std::uint64_t n_cols);
std::unique_ptr<treelite::Model> fit_rf_regressor(raft::handle_t const& handle, Device2DArrayView X,
    Device1DArrayView y, std::uint32_t n_trees, std::uint32_t max_depth);

}  // namespace fil_bench

#endif  // FIL_BENCH_DATAGEN_HPP_
