#include <fil_bench/datagen.hpp>
#include <fil_bench/array_types.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/make_regression.cuh>
#include <treelite/tree.h>
#include <cuml/ensemble/randomforest.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace fil_bench {

std::pair<Device2DArray, Device1DArray> make_regression(const raft::handle_t& handle,
    std::uint64_t n_rows, std::uint64_t n_cols) {
  auto [X, y] = make_empty(handle, n_rows, n_cols);

  raft::random::make_regression(handle,
      X.data_handle(),
      y.data_handle(),
      n_rows,
      n_cols,
      n_cols / 3,
      handle.get_stream(),
      (float*)nullptr,
      std::uint64_t(1),
      0.0f,
      n_cols / 3,
      0.1f,
      0.01f,
      false,
      12345ULL);
  handle.sync_stream();
  handle.sync_stream_pool();

  return {X, y};
}

std::pair<Device2DArray, Device1DArray> make_empty(
    raft::handle_t const& handle, std::uint64_t n_rows, std::uint64_t n_cols) {
  Device2DArray X = raft::make_device_matrix<float>(handle, n_rows, n_cols);
  Device1DArray y = raft::make_device_vector<float>(handle, n_rows);
  return {X, y};
}

std::unique_ptr<treelite::Model> fit_rf_regressor(
    const raft::handle_t& handle, Device2DArrayView X, Device1DArrayView y,
    std::uint32_t n_trees, std::uint32_t max_depth) {
  // Take first 1000 rows as training set
  auto train_nrows = std::min(X.extent(0), std::uint64_t(1000));
  auto rf_model = std::make_unique<ML::RandomForestRegressorF>();
  auto* rf_model_ptr = rf_model.get();
  ML::RF_params rf_params = ML::set_rf_params(
      static_cast<int>(max_depth),
      (1 << 20),
      1.f,
      32,
      3,
      3,
      0.0f,
      true,
      static_cast<int>(n_trees),
      1.f,
      1234ULL,
      ML::CRITERION::MSE,
      8,
      128
  );
  ML::fit(handle, rf_model_ptr, X.data_handle(), train_nrows, X.extent(1), y.data_handle(), rf_params);
  handle.sync_stream();
  handle.sync_stream_pool();

  void* tl_model_ptr{nullptr};
  ML::build_treelite_forest(&tl_model_ptr, rf_model.get(), X.extent(1));

  return std::unique_ptr<treelite::Model>{static_cast<treelite::Model*>(tl_model_ptr)};
}

}  // namespace fil_bench
