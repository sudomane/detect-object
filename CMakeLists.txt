cmake_minimum_required (VERSION 3.14)
project (OBJ_DETECTION CUDA CXX)

option(PROF "Reference build" NO)
include(FetchContent)
include(CheckLanguage)
check_language(CUDA)

FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )

FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_executable(main src/main.cu
                    src/error.cu
                    src/img_io.cpp
                    src/img_grayscale.cu
                    src/img_conv_2D.cu
                    src/img_diff.cu
                    src/test_grayscale.cu
                    src/test_conv_2D.cu
                    src/test_diff.cu
                    src/test_open.cu
                    src/img_morph.cu
                    src/test_morph.cu
                    src/utils.cpp
                    src/utils.h)

target_compile_features(main PUBLIC cxx_std_17)
target_link_libraries(main PRIVATE PNG::PNG CLI11::CLI11 spdlog::spdlog)

#add_executable(bench src/bench.cpp)
#target_compile_features(bench PUBLIC cxx_std_17)
#target_link_libraries(bench PRIVATE spdlog::spdlog benchmark::benchmark)



