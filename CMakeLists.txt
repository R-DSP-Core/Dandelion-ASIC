cmake_minimum_required(VERSION 3.16)

project(DandelionAsicNodes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dandelion_asic
    simulator/src/protocol.cpp
    simulator/src/tensor_node.cpp
    simulator/src/spatial_node.cpp
)

target_include_directories(dandelion_asic PUBLIC simulator/include)
target_compile_options(dandelion_asic PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(asic_tests simulator/tests/test_nodes.cpp)
target_link_libraries(asic_tests PRIVATE dandelion_asic)
target_compile_options(asic_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_test(NAME asic_nodes COMMAND asic_tests)
