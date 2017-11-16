# generate so
set(gpu_src math_util.cu)
cuda_add_library(gpu SHARED ${gpu_src})
target_link_libraries(gpu ${LINKER_LIBS})