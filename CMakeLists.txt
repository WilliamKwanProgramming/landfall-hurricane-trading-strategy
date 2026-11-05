# Root build file: the C++ footprint kernel, its C++ test and the pybind11 module landfall_cpp.
#   cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
# Options: LANDFALL_SANITIZE=ON (ASan + UBSan; the Python module is then skipped because the
# interpreter is not instrumented), LANDFALL_WERROR=ON.
cmake_minimum_required(VERSION 3.16)
project(landfall LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LANDFALL_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(LANDFALL_WERROR "Treat compiler warnings as errors" OFF)

add_library(landfall_options INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(landfall_options INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
  if(LANDFALL_WERROR)
    target_compile_options(landfall_options INTERFACE -Werror)
  endif()
  if(LANDFALL_SANITIZE)
    target_compile_options(landfall_options INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    target_link_options(landfall_options INTERFACE -fsanitize=address,undefined)
  endif()
endif()

enable_testing()
add_subdirectory(cpp)
