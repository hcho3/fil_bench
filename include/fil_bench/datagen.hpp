#ifndef FIL_BENCH_DATAGEN_HPP_
#define FIL_BENCH_DATAGEN_HPP_

#include <memory>
#include <utility>

#include <fil_bench/array_types.hpp>
#include <fil_bench/constants.hpp>

namespace raft {

class handle_t;

}  // namespace raft

namespace treelite {

class Model;

}  // namespace treelite

namespace fil_bench {

std::pair<Device2DArray, Device1DArray> make_regression(raft::handle_t const& handle);
std::unique_ptr<treelite::Model> fit_rf_regressor(
    raft::handle_t const& handle, Device2DArrayView X, Device1DArrayView y);

}  // namespace fil_bench

#endif  // FIL_BENCH_DATAGEN_HPP_
