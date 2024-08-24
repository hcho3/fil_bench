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

add_executable(tuner)
target_sources(tuner
    PRIVATE
    include/fil_bench/launch_config.hpp
    include/fil_bench/tuner.hpp
    src/launch_config.cpp
    src/tuner.cpp
    src/main.cpp
)
target_include_directories(tuner
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
)
target_link_libraries(tuner PRIVATE cuml::cuml++ raft::raft treelite::treelite argparse)
set_target_properties(tuner
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
