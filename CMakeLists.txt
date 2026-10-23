cmake_minimum_required(VERSION 3.16)
project(neuralplsi LANGUAGES CXX)

# The C++ engine. `pip install .` builds it through scikit-build-core and puts
# the library inside the Python package; `cmake -S . -B build` builds it and
# the example on their own.
if(SKBUILD)
  set(NEURALPLSI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
endif()
if(NOT SKBUILD)
  enable_testing()
endif()
add_subdirectory(core)

if(SKBUILD)
  install(TARGETS neuralplsi
          LIBRARY DESTINATION neuralplsi/_lib
          RUNTIME DESTINATION neuralplsi/_lib)
endif()
