#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <fil_bench/datagen.hpp>
#include <fil_bench/raft_handle.hpp>
#include <fil_bench/tuner.hpp>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/serialize.hpp>
#include <treelite/detail/file_utils.h>
#include <treelite/tree.h>

std::filesystem::path validate_directory_path(std::string const& str) {
  auto path = std::filesystem::weakly_canonical(std::filesystem::u8path(str));
  TREELITE_CHECK(std::filesystem::exists(path)) << "Path " << path << " does not exist";
  TREELITE_CHECK(std::filesystem::is_directory(path)) << "Path " << path << " must be a directory";
  return path;
}

int main(int argc, char* argv[]) {
  std::string outdir_str;
  std::uint64_t n_rows, n_cols;
  std::uint32_t n_trees, max_depth, n_reps;

  argparse::ArgumentParser argparser("tuner");
  argparser.add_argument("--outdir")
      .required()
      .help("Path to output dir")
      .metavar("OUTDIR")
      .store_into(outdir_str);
  argparser.add_argument("--n_rows")
      .default_value(std::uint64_t(1e6))
      .help("Number of rows in the generated data")
      .metavar("N_ROWS")
      .store_into(n_rows);
  argparser.add_argument("--n_cols")
      .default_value(std::uint64_t(20))
      .help("Number of columns in the generated data")
      .metavar("N_COLS")
      .store_into(n_cols);
  argparser.add_argument("--n_trees")
      .default_value(std::uint32_t(1000))
      .help("Number of trees in the generated tree ensemble model")
      .metavar("N_TREES")
      .store_into(n_trees);
  argparser.add_argument("--max_depth")
      .default_value(std::uint32_t(10))
      .help("Maximum depth of each tree in the generated tree ensemble model")
      .metavar("MAX_DEPTH")
      .store_into(max_depth);
  argparser.add_argument("--n_reps")
      .default_value(std::uint32_t(10))
      .help("Number of times to run each kernel configuration to measure its performance")
      .metavar("N_REPS")
      .store_into(n_reps);
  try {
    argparser.parse_args(argc, argv);
  } catch (std::runtime_error const& e) {
    std::cerr << e.what() << "\n" << std::endl;
    std::cerr << argparser.help().str() << std::endl;
    return -1;
  }

  auto outdir = validate_directory_path(outdir_str);
  auto model_path = outdir / "model.tl";
  auto data_path = outdir / "X.npy";
  auto result_path = outdir / "tune_result.json";

  std::cout << "outdir = " << outdir << ", n_rows = " << n_rows << ", n_cols = " << n_cols
            << ", n_trees = " << n_trees << ", max_depth = " << max_depth << ", n_reps = " << n_reps
            << std::endl;

  raft::handle_t handle = fil_bench::make_raft_handle();
  auto [X, y] = fil_bench::make_regression(handle, n_rows, n_cols);
  {
    std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream(data_path);
    raft::serialize_mdspan(handle, ofs, X.view());
    handle.sync_stream();
    handle.sync_stream_pool();
  }

  std::cout << "Generated synthetic data of dimensions (" << n_rows << ", " << n_cols << "). "
            << "Fitting a random forest..." << std::endl;
  auto rf_model = fil_bench::fit_rf_regressor(handle, X.view(), y.view(), n_trees, max_depth);
  {
    std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream(model_path);
    rf_model->SerializeToStream(ofs);
  }

  std::cout << "Fitted a random forest. Tuning parameters for FIL..." << std::endl;
  auto [best_config_old_fil, best_time_old_fil]
      = fil_bench::optimize_old_fil(handle, rf_model.get(), X.view(), n_reps);
  std::cout << "Best configuration for old FIL: " << best_config_old_fil
            << ", time elapsed = " << (static_cast<double>(best_time_old_fil) / 1000000) << " ms"
            << std::endl;
  auto [best_config_new_fil, best_time_new_fil]
      = fil_bench::optimize_new_fil(handle, rf_model.get(), X.view(), n_reps);
  std::cout << "Best configuration for new FIL: " << best_config_new_fil
            << ", time elapsed = " << (static_cast<double>(best_time_new_fil) / 1000000) << " ms"
            << std::endl;

  {
    std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream(result_path);
    nlohmann::ordered_json result{{"old_fil", best_config_old_fil},
        {"new_fil", best_config_new_fil},
        {"metadata", {{"model", model_path}, {"data", data_path}, {"n_rows", n_rows},
                         {"n_cols", n_cols}, {"n_trees", n_trees}, {"max_depth", max_depth}}}};
    ofs << result.dump(4) << std::endl;
  }

  return 0;
}
