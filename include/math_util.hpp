#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cblas.h>
#include <cmath>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include "glog/logging.h"


#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: various checks for different function calls.
/* Code block avoids redefinition of cudaError_t error */
#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

const char* cublasGetErrorString(cublasStatus_t error){
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

namespace tensorrt{

    template <typename Dtype>
    __global__ void set_kernel(const int n, const Dtype alpha, Dtype* y);

    template <typename Dtype>
    void tensorrt_gpu_set(const int N, const Dtype alpha, Dtype* Y);

    // CUDA: number of blocks for threads.
    inline int CAFFE_GET_BLOCKS(const int N) {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }

    template <typename Dtype>
    __global__ void powx_kernel(const int n, const Dtype* a, const Dtype alpha, Dtype* y);

    template <typename Dtype>
    void tensorrt_gpu_powx(const int N, const Dtype* a,
                           const Dtype alpha, Dtype* y);

    /**
     * y = sum(abs(x))
     * @tparam Dtype
     * @param cublas_handle_
     * @param n
     * @param x
     * @param y
     */
    template <typename Dtype>
    void tensorrt_gpu_asum(cublasHandle_t cublas_handle_, const int n, const Dtype* x, Dtype* y);

    template <typename Dtype>
    void tensorrt_gpu_scale(cublasHandle_t cublas_handle_, const int n, const Dtype alpha, const Dtype *x,
                            Dtype* y);

    template <typename Dtype>
    void tensorrt_gpu_divbsx(const int num, const Dtype* A,
                             const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                             Dtype* B);
    /**
    A : rows * cols
    num : in fact number of elements
    v : vectors
    notrans: b[r][c] = a[r][c] * v[c]
    trans: b[r][c] = a[r][c] * v[r]
    **/
    template <typename Dtype>
    void tensorrt_gpu_mulbsx(const int num, const Dtype* A,
                             const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                             Dtype* B);
    /**
    A : M * N
    x : M or N
    y : N or M
    y = alpha * A * x + beta * y
    if not transA, x length is N, y length is M
    in fact, y[i] = alpha * sum(a[i][k] * x[k]) + beta * y[i], k is from 0 to N
    else, x length is M, y length is N
    y[i] = alpha * sum(a[k][i] * x[k]) + beta * y[i], k is from 0 to M
    **/
    template <typename Dtype>
    void tensorrt_gpu_gemv(cublasHandle_t cublas_handle_, const CBLAS_TRANSPOSE TransA, const int M, const int N,
                           const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
                           Dtype* y);

    /**
    **/
    template <typename Dtype>
    __global__ void PermuteKernel(const int nthreads, Dtype* const bottom_data, const bool forward,
                                  const int* permute_order, const int* old_steps, const int* new_steps,
                                  const int num_axes, Dtype* const top_data);

    template <typename Dtype>
    void tensorrt_gpu_permute(const int count, Dtype* const bottom_data, const int* permute_order,
                              const int* old_steps, const int* new_steps, const int num_axes, Dtype* const top_data);

}