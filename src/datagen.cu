#include <fil_bench/datagen.hpp>
#include <fil_bench/array_types.hpp>
#include <fil_bench/constants.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/make_regression.cuh>
#include <treelite/tree.h>
#include <cuml/ensemble/randomforest.hpp>

#include <memory>
#include <utility>

namespace fil_bench {

std::pair<Device2DArray, Device1DArray> make_regression(const raft::handle_t& handle) {
  Device2DArray X = raft::make_device_matrix<float>(handle, nrows, ncols);
  auto y = raft::make_device_vector<float>(handle, nrows);

  raft::random::make_regression(handle,
      X.data_handle(),
      y.data_handle(),
      nrows,
      ncols,
      6,
      handle.get_stream(),
      (float*)nullptr,
      1,
      0.0f,
      6,
      0.1f,
      0.01f,
      false,
      12345ULL);
  handle.sync_stream();
  handle.sync_stream_pool();

  return {X, y};
}

std::unique_ptr<treelite::Model> fit_rf_regressor(
    const raft::handle_t& handle, Device2DArrayView X, Device1DArrayView y) {
  // Take first 1000 rows as training set
  auto train_nrows = std::min(nrows, 1000);
  auto rf_model = std::make_unique<ML::RandomForestRegressorF>();
  auto* rf_model_ptr = rf_model.get();
  ML::RF_params rf_params = ML::set_rf_params(10,
      (1 << 20),
      1.f,
      32,
      3,
      3,
      0.0f,
      true,
      1,
      1.f,
      1234ULL,
      ML::CRITERION::MSE,
      8,
      128
  );
  ML::fit(handle, rf_model_ptr, X.data_handle(), train_nrows, ncols, y.data_handle(), rf_params);
  handle.sync_stream();
  handle.sync_stream_pool();

  void* tl_model_ptr{nullptr};
  ML::build_treelite_forest(&tl_model_ptr, rf_model.get(), ncols);

  return std::unique_ptr<treelite::Model>{static_cast<treelite::Model*>(tl_model_ptr)};
}

}  // namespace fil_bench
