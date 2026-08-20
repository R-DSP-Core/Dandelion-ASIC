cmake_minimum_required(VERSION 3.16)

project(DandelionAsicNodes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dandelion_asic_abi INTERFACE)
target_include_directories(dandelion_asic_abi INTERFACE simulator/include)

add_library(dandelion_asic_protocol simulator/src/protocol.cpp)
target_link_libraries(dandelion_asic_protocol PUBLIC dandelion_asic_abi)
target_compile_options(dandelion_asic_protocol PRIVATE -Wall -Wextra -Wpedantic -Werror)

# C++ nodes are a differential-reference implementation. The RTL simulator
# links only dandelion_asic_protocol, which keeps model execution out of that
# path while retaining the descriptor/image serializer as a shared ABI helper.
add_library(dandelion_asic
    simulator/src/tensor_node.cpp
    simulator/src/spatial_node.cpp
)

target_link_libraries(dandelion_asic PUBLIC dandelion_asic_protocol)
target_compile_options(dandelion_asic PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(asic_tests simulator/tests/test_nodes.cpp)
target_link_libraries(asic_tests PRIVATE dandelion_asic)
target_compile_options(asic_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_test(NAME asic_nodes COMMAND asic_tests)

find_program(SBT_EXECUTABLE NAMES sbt REQUIRED)
file(GLOB_RECURSE DANDELION_ACCELERATOR_RTL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/RTL/src/main/scala/*.scala")
set(DANDELION_ACCELERATOR_RTL_DIR "${CMAKE_CURRENT_BINARY_DIR}/rtl")
set(DANDELION_ACCELERATOR_RTL_SV
    "${DANDELION_ACCELERATOR_RTL_DIR}/DandelionAccelerators.sv")

# Verilator constructs its generated build graph during CMake configuration,
# so the source must exist before the parent simulator invokes verilate().
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${DANDELION_ACCELERATOR_RTL_DIR}"
)
execute_process(
    COMMAND "${SBT_EXECUTABLE}" "runMain dandelion.asic.Elaborate ${DANDELION_ACCELERATOR_RTL_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/RTL"
    RESULT_VARIABLE DANDELION_ACCELERATOR_RTL_RESULT
    OUTPUT_VARIABLE DANDELION_ACCELERATOR_RTL_OUTPUT
    ERROR_VARIABLE DANDELION_ACCELERATOR_RTL_ERROR
)
if(NOT DANDELION_ACCELERATOR_RTL_RESULT EQUAL 0)
    message(FATAL_ERROR
        "failed to elaborate Dandelion accelerator RTL:\n"
        "${DANDELION_ACCELERATOR_RTL_OUTPUT}${DANDELION_ACCELERATOR_RTL_ERROR}")
endif()

add_custom_command(
    OUTPUT "${DANDELION_ACCELERATOR_RTL_SV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${DANDELION_ACCELERATOR_RTL_DIR}"
    COMMAND "${SBT_EXECUTABLE}" "runMain dandelion.asic.Elaborate ${DANDELION_ACCELERATOR_RTL_DIR}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/RTL"
    DEPENDS ${DANDELION_ACCELERATOR_RTL_SOURCES}
            "${CMAKE_CURRENT_SOURCE_DIR}/RTL/build.sbt"
            "${CMAKE_CURRENT_SOURCE_DIR}/RTL/project/build.properties"
    VERBATIM
)
add_custom_target(dandelion_asic_rtl DEPENDS "${DANDELION_ACCELERATOR_RTL_SV}")

# Export the generated RTL location to the simulator parent directory.
set(DANDELION_ACCELERATOR_RTL_SV "${DANDELION_ACCELERATOR_RTL_SV}" PARENT_SCOPE)
