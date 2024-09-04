#include <filesystem>
#include <fstream>

#include <fil_bench/constants.hpp>
#include <fil_bench/datagen.hpp>
#include <fil_bench/raft_handle.hpp>
#include <fil_bench/tuner.hpp>

#include <argparse/argparse.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/serialize.hpp>
#include <treelite/detail/file_utils.h>
#include <treelite/tree.h>

namespace {

constexpr int predict_repetitions = 10;

}  // anonymous namespace

int main() {
  raft::handle_t handle = fil_bench::make_raft_handle();
  auto [X, y] = fil_bench::make_regression(handle);
  auto rf_model = fil_bench::fit_rf_regressor(handle, X.view(), y.view());
  {
    std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream("./model.tl");
    rf_model->SerializeToStream(ofs);
  }
  {
    std::ofstream ofs = treelite::detail::OpenFileForWriteAsStream("./X.npy");
    raft::serialize_mdspan(handle, ofs, X.view());
    handle.sync_stream();
    handle.sync_stream_pool();
  }

  auto best_config_old_fil = fil_bench::optimize_old_fil(handle, rf_model.get(), X.view());
  std::cout << "Best configuration for old FIL: " << best_config_old_fil << std::endl;
  auto best_config_new_fil = fil_bench::optimize_new_fil(handle, rf_model.get(), X.view());
  std::cout << "Best configuration for new FIL: " << best_config_new_fil << std::endl;

  return 0;
}
