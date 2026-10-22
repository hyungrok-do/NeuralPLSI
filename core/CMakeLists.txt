cmake_minimum_required(VERSION 3.16)
project(neuralplsi_core VERSION 0.1.0 LANGUAGES CXX)

option(NEURALPLSI_BUILD_EXAMPLES "Build the C++ example and register it as a test" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Threads REQUIRED)

# One shared library with a plain C ABI (include/neuralplsi.h), loaded by the
# Python package through ctypes. The R package and the WebAssembly build
# compile the same two sources with their own toolchains.
add_library(neuralplsi SHARED src/neuralplsi.cpp src/spline.cpp)
target_include_directories(neuralplsi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(neuralplsi PUBLIC cxx_std_17)
set_target_properties(neuralplsi PROPERTIES CXX_EXTENSIONS OFF PREFIX "lib")
target_link_libraries(neuralplsi PRIVATE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # The flags of the Python source-checkout build. Fused multiply-add stays off,
  # so every build on a machine produces the same float32 arithmetic.
  target_compile_options(neuralplsi PRIVATE -O3 -ffp-contract=off -Wall -Wextra)
elseif(MSVC)
  target_compile_options(neuralplsi PRIVATE /O2 /fp:precise /utf-8)
  set_target_properties(neuralplsi PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NEURALPLSI_BUILD_EXAMPLES)
  enable_testing()
  add_executable(neuralplsi_example examples/native_neuralplsi.cpp)
  target_link_libraries(neuralplsi_example PRIVATE neuralplsi)
  set_target_properties(neuralplsi_example PROPERTIES CXX_EXTENSIONS OFF)
  add_test(NAME example COMMAND neuralplsi_example ${CMAKE_CURRENT_BINARY_DIR}/example.nplsi)
endif()
