#include <ostream>

#include <fil_bench/launch_config.hpp>

#include <nlohmann/json.hpp>

namespace ML::fil {

NLOHMANN_JSON_SERIALIZE_ENUM(algo_t,
    {{algo_t::ALGO_AUTO, "ML::fil::algo_t::ALGO_AUTO"}, {algo_t::NAIVE, "ML::fil::algo_t::NAIVE"},
        {algo_t::TREE_REORG, "ML::fil::algo_t::TREE_REORG"},
        {algo_t::BATCH_TREE_REORG, "ML::fil::algo_t::BATCH_TREE_REORG"}})

NLOHMANN_JSON_SERIALIZE_ENUM(
    storage_type_t, {{storage_type_t::AUTO, "ML::fil::storage_type_t::AUTO"},
                        {storage_type_t::DENSE, "ML::fil::storage_type_t::DENSE"},
                        {storage_type_t::SPARSE, "ML::fil::storage_type_t::SPARSE"},
                        {storage_type_t::SPARSE8, "ML::fil::storage_type_t::SPARSE8"}})

}  // namespace ML::fil

namespace ML::experimental::fil {

NLOHMANN_JSON_SERIALIZE_ENUM(tree_layout,
    {{tree_layout::depth_first, "ML::experimental::fil::tree_layout::depth_first"},
        {tree_layout::breadth_first, "ML::experimental::fil::tree_layout::breadth_first"}})

}  // namespace ML::experimental::fil

namespace fil_bench {

std::ostream& operator<<(std::ostream& os, launch_config_t const& config) {
  os << "{chunk_size=" << config.chunk_size << ", algo_type=";
  switch (config.algo_type) {
  case ML::fil::algo_t::ALGO_AUTO:
    os << "ML::fil::algo_t::ALGO_AUTO, ";
    break;
  case ML::fil::algo_t::NAIVE:
    os << "ML::fil::algo_t::NAIVE, ";
    break;
  case ML::fil::algo_t::TREE_REORG:
    os << "ML::fil::algo_t::TREE_REORG, ";
    break;
  case ML::fil::algo_t::BATCH_TREE_REORG:
    os << "ML::fil::algo_t::BATCH_TREE_REORG, ";
    break;
  }

  os << "storage_type=";
  switch (config.storage_type) {
  case ML::fil::storage_type_t::AUTO:
    os << "ML::fil::storage_type_t::AUTO, ";
    break;
  case ML::fil::storage_type_t::DENSE:
    os << "ML::fil::storage_type_t::DENSE, ";
    break;
  case ML::fil::storage_type_t::SPARSE:
    os << "ML::fil::storage_type_t::SPARSE, ";
    break;
  case ML::fil::storage_type_t::SPARSE8:
    os << "ML::fil::storage_type_t::SPARSE8, ";
    break;
  }

  os << "layout=";
  switch (config.layout) {
  case ML::experimental::fil::tree_layout::depth_first:
    os << "ML::experimental::fil::tree_layout::depth_first}";
    break;
  case ML::experimental::fil::tree_layout::breadth_first:
    os << "ML::experimental::fil::tree_layout::breadth_first}";
    break;
  }

  return os;
}

void to_json(nlohmann::ordered_json& js, launch_config_t const& config) {
  js = nlohmann::ordered_json{{"chunk_size", config.chunk_size}, {"algo_type", config.algo_type},
      {"storage_type", config.storage_type}, {"layout", config.layout}};
}

void from_json(nlohmann::ordered_json const& js, launch_config_t& config) {
  js.at("chunk_size").get_to(config.chunk_size);
  js.at("algo_type").get_to(config.algo_type);
  js.at("storage_type").get_to(config.storage_type);
  js.at("layout").get_to(config.layout);
}

}  // namespace fil_bench
