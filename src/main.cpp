#include <fstream>

#include <fil_bench/tuner.hpp>

#include <argparse/argparse.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/serialize.hpp>
#include <treelite/detail/file_utils.h>
#include <treelite/tree.h>

namespace {

constexpr std::uint64_t nrows = 1000000;
constexpr std::uint64_t ncols = 20;

constexpr int predict_repetitions = 10;

}  // anonymous namespace

int main() {
  raft::handle_t handle;
  fil_bench::Device2DArray X = raft::make_device_matrix<float>(handle, nrows, ncols);

  {
    std::ifstream ifs = treelite::detail::OpenFileForReadAsStream("./X.npy");
    raft::deserialize_mdspan(handle, ifs, X.view());
  }

  std::unique_ptr<treelite::Model> tl_model;
  {
    std::ifstream ifs = treelite::detail::OpenFileForReadAsStream("./model.tl");
    tl_model = treelite::Model::DeserializeFromStream(ifs);
  }

  auto best_config_old_fil = fil_bench::optimize_old_fil(handle, tl_model.get(), X.view());
  std::cout << "Best configuration for old FIL: " << best_config_old_fil << std::endl;
  auto best_config_new_fil = fil_bench::optimize_new_fil(handle, tl_model.get(), X.view());
  std::cout << "Best configuration for new FIL: " << best_config_new_fil << std::endl;

  return 0;
}
