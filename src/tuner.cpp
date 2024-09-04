#include <limits>
#include <vector>

#include <fil_bench/constants.hpp>
#include <fil_bench/launch_config.hpp>
#include <fil_bench/tuner.hpp>

#include <cuml/experimental/fil/treelite_importer.hpp>
#include <cuml/fil/fil.h>
#include <treelite/tree.h>

namespace {

constexpr int predict_repetitions = 10;

}  // anonymous namespace

namespace fil_bench {

std::pair<launch_config_t, std::int64_t> optimize_old_fil(
    raft::handle_t& handle, treelite::Model* tl_model, Device2DArrayView X) {
  launch_config_t best_config = {
      .layout = ML::experimental::fil::tree_layout::breadth_first,
  };
  auto min_time = std::numeric_limits<std::int64_t>::max();
  auto ypred = raft::make_device_vector<float>(handle, X.extent(0));

  ML::fil::treelite_params_t tl_params = {.algo = ML::fil::algo_t::NAIVE,
      .output_class = false,
      .threshold = 1.f,
      .storage_type = ML::fil::storage_type_t::DENSE,
      .blocks_per_sm = 8,
      .threads_per_tree = 1,
      .n_items = 0,
      .pforest_shape_str = nullptr};

  ML::fil::forest_t<float> forest;
  ML::fil::forest_variant forest_variant;

  auto allowed_storage_types = std::vector<ML::fil::storage_type_t>{ML::fil::storage_type_t::DENSE,
      ML::fil::storage_type_t::SPARSE, ML::fil::storage_type_t::SPARSE8};

  for (auto storage_type : allowed_storage_types) {
    auto allowed_algo_types = std::vector<ML::fil::algo_t>{ML::fil::algo_t::NAIVE};
    if (storage_type == ML::fil::storage_type_t::DENSE) {
      allowed_algo_types.push_back(ML::fil::algo_t::TREE_REORG);
      allowed_algo_types.push_back(ML::fil::algo_t::BATCH_TREE_REORG);
    }
    tl_params.storage_type = storage_type;

    for (auto algo_type : allowed_algo_types) {
      tl_params.algo = algo_type;

      for (auto chunk_size = 1; chunk_size <= 32; chunk_size *= 2) {
        tl_params.threads_per_tree = chunk_size;
        ML::fil::from_treelite(handle, &forest_variant, tl_model, &tl_params);
        forest = std::get<ML::fil::forest_t<float>>(forest_variant);

        auto tstart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < predict_repetitions; i++) {
          ML::fil::predict(
              handle, forest, ypred.data_handle(), X.data_handle(), X.extent(0), false);
        }
        handle.sync_stream();
        handle.sync_stream_pool();

        auto tend = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
        if (elapsed < min_time) {
          min_time = elapsed;
          best_config.chunk_size = chunk_size;
          best_config.storage_type = storage_type;
          best_config.algo_type = algo_type;
        }
      }
    }
  }

  return {best_config, min_time};
}

std::pair<launch_config_t, std::int64_t> optimize_new_fil(
    raft::handle_t& handle, treelite::Model* tl_model, Device2DArrayView X) {
  launch_config_t best_config
      = {.algo_type = ML::fil::algo_t::NAIVE, .storage_type = ML::fil::storage_type_t::DENSE};
  auto min_time = std::numeric_limits<std::int64_t>::max();
  auto ypred = raft::make_device_vector<float>(handle, X.extent(0));

  auto allowed_layouts = std::vector<ML::experimental::fil::tree_layout>{
      ML::experimental::fil::tree_layout::breadth_first,
      ML::experimental::fil::tree_layout::depth_first,
  };

  for (auto layout : allowed_layouts) {
    auto filex_model = ML::experimental::fil::import_from_treelite_handle(
        tl_model, layout, 128, false, raft_proto::device_type::gpu, 0, handle.get_stream());
    handle.sync_stream();
    handle.sync_stream_pool();

    for (auto chunk_size = 1; chunk_size <= 32; chunk_size *= 2) {
      auto tstart = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < predict_repetitions; i++) {
        filex_model.predict(handle, ypred.data_handle(), X.data_handle(), X.extent(0),
            raft_proto::device_type::gpu, raft_proto::device_type::gpu,
            ML::experimental::fil::infer_kind::default_kind, chunk_size);
      }
      handle.sync_stream();
      handle.sync_stream_pool();

      auto tend = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
      if (elapsed < min_time) {
        min_time = elapsed;
        best_config.chunk_size = chunk_size;
        best_config.layout = layout;
      }
    }
  }

  return {best_config, min_time};
}

}  // namespace fil_bench
