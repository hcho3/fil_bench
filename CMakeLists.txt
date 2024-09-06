cmake_minimum_required(VERSION 3.28)
project(fil_bench LANGUAGES C CXX)

find_package(cuml 24.10 CONFIG REQUIRED)
find_package(raft 24.10 CONFIG REQUIRED)
find_package(Treelite 4.3.0 CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v3.1
)
FetchContent_MakeAvailable(argparse)
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(fil_bench OBJECT)
target_sources(fil_bench
    PRIVATE
    include/fil_bench/array_types.hpp
    include/fil_bench/constants.hpp
    include/fil_bench/datagen.hpp
    include/fil_bench/fwd_decl.hpp
    include/fil_bench/launch_config.hpp
    include/fil_bench/raft_handle.hpp
    include/fil_bench/tuner.hpp
    src/datagen.cu
    src/launch_config.cpp
    src/raft_handle.cpp
    src/tuner.cpp
)
target_include_directories(fil_bench
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)
target_link_libraries(fil_bench
    PRIVATE
    cuml::cuml++
    PUBLIC
    raft::raft
    treelite::treelite
    nlohmann_json::nlohmann_json)

add_executable(tuner src/tuner_cli.cpp)
target_link_libraries(tuner
    PRIVATE
    fil_bench
    argparse::argparse)

add_executable(runner src/runner_cli.cpp)
target_link_libraries(runner
    PRIVATE
    fil_bench
    argparse::argparse)

foreach(target fil_bench tuner runner)
    set_target_properties(${target}
        PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
endforeach()
