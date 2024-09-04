#include <utility>

#include <fil_bench/constants.hpp>
#include <fil_bench/raft_handle.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_pool.hpp>

namespace fil_bench {

raft::handle_t make_raft_handle() {
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(num_streams);
  return raft::handle_t{rmm::cuda_stream_per_thread, stream_pool};
}

}  // namespace fil_bench
