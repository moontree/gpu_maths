cmake_minimum_required(VERSION 3.4)
project(gpu_maths)
set(LINKER_LIBS "")

include_directories(${PROJECT_SOURCE_DIR}/include)
message(${PROJECT_SOURCE_DIR})
# cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBS}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
# cublas and curand
list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
message(STATUS "  lib  extend : ${LINKER_LIBS}")
# glog
include(${PROJECT_SOURCE_DIR}/cmake/Modules/FindGlog.cmake)
message(STATUS "glog library status:")
message(STATUS "    version: ${GLOG_VERSION}")
message(STATUS "    libraries: ${GLOG_LIBRARY}")
message(STATUS "    include path: ${GLOG_INCLUDE_DIRS}")
# include dirs
include_directories(${CUDA_INCLUDE_DIRS} ${GLOG_INCLUDE_DIRS})
# set outout
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
# add sub directory
add_subdirectory(src)
add_subdirectory(apps)

