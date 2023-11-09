cmake_minimum_required(VERSION 3.12)
project(qwen.cpp VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif ()

# third-party libraries
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)

add_subdirectory(third_party/re2)

add_compile_definitions(GGML_CUDA_MMV_Y=3)  # for large vocab
include_directories(third_party/ggml/include/ggml third_party/ggml/src)
add_subdirectory(third_party/ggml)

if(APPLE AND GGML_METAL)
  add_compile_definitions(GGML_USE_METAL)
  configure_file(third_party/ggml/src/ggml-metal.metal ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)
endif()

if (GGML_CUBLAS)
  add_compile_definitions(GGML_USE_CUBLAS)
  set(CUDA_ARCHITECTURES "52;61;70;75;80;86" CACHE STRING "qwen: cuda architectures to compile")
  set_property(TARGET ggml PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
endif ()

file(GLOB CPP_SOURCES
  ${PROJECT_SOURCE_DIR}/*.h
  ${PROJECT_SOURCE_DIR}/*.cpp)

set_source_files_properties(${CPP_SOURCES} PROPERTIES COMPILE_FLAGS "-pedantic-errors")

add_library(qwen STATIC qwen.cpp)
target_link_libraries(qwen PUBLIC re2::re2 ggml)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE qwen)

# GoogleTest
option(QWEN_ENABLE_TESTING "qwen: enable testing" OFF)
if (QWEN_ENABLE_TESTING)
  enable_testing()

  # ref: https://github.com/google/googletest/blob/main/googletest/README.md
  include(FetchContent)
  FetchContent_Declare(
    googletest
    # Specify the commit you depend on and update it regularly.
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)

  # Now simply link against gtest or gtest_main as needed. Eg
  add_executable(qwen_test qwen_test.cpp)
  target_link_libraries(qwen_test PRIVATE qwen gtest_main)
  gtest_discover_tests(qwen_test)
endif ()

option(QWEN_ENABLE_PYBIND, "qwen: enable python binding" OFF)
if (QWEN_ENABLE_PYBIND)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(qwen ggml PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  add_subdirectory(third_party/pybind11)
  pybind11_add_module(_C qwen_pybind.cpp)
  target_link_libraries(_C PRIVATE qwen)
endif ()

# lint
file(GLOB PY_SOURCES
  ${PROJECT_SOURCE_DIR}/qwen_cpp/*.py
  ${PROJECT_SOURCE_DIR}/tiktoken_cpp/*.py
  ${PROJECT_SOURCE_DIR}/setup.py)
add_custom_target(lint
  COMMAND clang-format -i ${CPP_SOURCES}
  COMMAND isort ${PY_SOURCES}
  COMMAND black ${PY_SOURCES} --line-length 120)
