#include <chrono>
#include <cstdint>

#include <fil_bench/launch_config.hpp>
#include <fil_bench/runner.hpp>

#include <cuml/experimental/fil/treelite_importer.hpp>
#include <cuml/fil/fil.h>
#include <raft/core/handle.hpp>
#include <treelite/tree.h>

namespace fil_bench {

std::int64_t run_old_fil(raft::handle_t& handle, launch_config_t launch_config,
    treelite::Model* rf_model, Device2DArrayView X, std::uint32_t n_reps) {
  ML::fil::treelite_params_t tl_params = {.algo = launch_config.algo_type,
      .output_class = false,
      .threshold = 1.f,
      .storage_type = launch_config.storage_type,
      .blocks_per_sm = 8,
      .threads_per_tree = launch_config.chunk_size,
      .n_items = 0,
      .pforest_shape_str = nullptr};

  ML::fil::forest_t<float> forest;
  ML::fil::forest_variant forest_variant;

  ML::fil::from_treelite(handle, &forest_variant, rf_model, &tl_params);
  forest = std::get<ML::fil::forest_t<float>>(forest_variant);

  auto ypred = raft::make_device_vector<float>(handle, X.extent(0));
  auto tstart = std::chrono::high_resolution_clock::now();
  for (std::uint32_t i = 0; i < n_reps; i++) {
    ML::fil::predict(handle, forest, ypred.data_handle(), X.data_handle(), X.extent(0), false);
  }
  handle.sync_stream();
  handle.sync_stream_pool();
  auto tend = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
}

std::int64_t run_new_fil(raft::handle_t& handle, launch_config_t launch_config,
    treelite::Model* rf_model, Device2DArrayView X, std::uint32_t n_reps) {
  auto filex_model = ML::experimental::fil::import_from_treelite_handle(rf_model,
      launch_config.layout, 128, false, raft_proto::device_type::gpu, 0, handle.get_stream());
  auto ypred = raft::make_device_vector<float>(handle, X.extent(0));

  auto tstart = std::chrono::high_resolution_clock::now();
  for (std::uint32_t i = 0; i < n_reps; i++) {
    filex_model.predict(handle, ypred.data_handle(), X.data_handle(), X.extent(0),
        raft_proto::device_type::gpu, raft_proto::device_type::gpu,
        ML::experimental::fil::infer_kind::default_kind, launch_config.chunk_size);
  }
  handle.sync_stream();
  handle.sync_stream_pool();

  auto tend = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
}

}  // namespace fil_bench
