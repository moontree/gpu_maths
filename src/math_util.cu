#include "math_util.hpp"




namespace tensorrt{
    cublasHandle_t cublas_handle_;
    // set_kernel
    template <typename Dtype>
    __global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
        CUDA_KERNEL_LOOP(index, n) {
            y[index] = alpha;
        }
    }

    template <typename Dtype>
    void tensorrt_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
      if (alpha == 0) {
        CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
        return;
      }
      // NOLINT_NEXT_LINE(whitespace/operators)
      set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, alpha, Y);
    }

    template void tensorrt_gpu_set<int>(const int N, const int alpha, int* Y);
    template void tensorrt_gpu_set<float>(const int N, const float alpha, float* Y);
    template void tensorrt_gpu_set<double>(const int N, const double alpha, double* Y);

    // power kernel
    template <typename Dtype>
    __global__ void powx_kernel(const int n, const Dtype* a, const Dtype alpha, Dtype* y) {
        CUDA_KERNEL_LOOP(index, n) {
            y[index] = pow(a[index], alpha);
        }
    }

    template <typename Dtype>
    void tensorrt_gpu_powx(const int N, const Dtype* a,
        const Dtype alpha, Dtype* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      powx_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, alpha, y);
    }

    template void tensorrt_gpu_powx<float>(const int N, const float* a,
        const float alpha, float* y);
    template void tensorrt_gpu_powx<double>(const int N, const double* a,
        const double alpha, double* y);

    // cublas handle as param, for temp
    template <>
    void tensorrt_gpu_asum<float>(cublasHandle_t cublas_handle_, const int n, const float* x, float* y) {
        CUBLAS_CHECK(cublasSasum(cublas_handle_, n, x, 1, y));
    }

    template <>
    void tensorrt_gpu_asum<double>(cublasHandle_t cublas_handle_, const int n, const double* x, double* y) {
        CUBLAS_CHECK(cublasDasum(cublas_handle_, n, x, 1, y));
    }

    template <>
    void tensorrt_gpu_scale<float>(cublasHandle_t cublas_handle_, const int n, const float alpha, const float *x,
                                float* y) {
        CUBLAS_CHECK(cublasScopy(cublas_handle_, n, x, 1, y, 1));
        CUBLAS_CHECK(cublasSscal(cublas_handle_, n, &alpha, y, 1));
    }

    template <>
    void tensorrt_gpu_scale<double>(cublasHandle_t cublas_handle_, const int n, const double alpha, const double *x,
                             double* y) {
        CUBLAS_CHECK(cublasDcopy(cublas_handle_, n, x, 1, y, 1));
        CUBLAS_CHECK(cublasDscal(cublas_handle_, n, &alpha, y, 1));
    }


    template <typename Dtype>
     __global__ void MulBsx(const int nthreads, const Dtype* A,
                           const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                           Dtype* B) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            int c = index % cols;
            int r = (index / cols) % rows;
            if (trans == CblasNoTrans) {
                B[index] = A[index] * v[c];
            } else {
                B[index] = A[index] * v[r];
            }
        }
    }

    template <typename Dtype>
    __global__ void DivBsx(const int nthreads, const Dtype* A,
                           const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                           Dtype* B) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            int c = index % cols;
            int r = (index / cols) % rows;
            if (trans == CblasNoTrans) {
                B[index] = A[index] / v[c];
            } else {
                B[index] = A[index] / v[r];
            }
        }
    }


    // divid a matrix with vector
    template <typename Dtype>
    void tensorrt_gpu_divbsx(const int num, const Dtype* A,
        const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
        Dtype* B) {
        DivBsx<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
            num, A, v, rows, cols, trans,
            B);
        CUDA_POST_KERNEL_CHECK;
    }



    // mult a matrix with vector
    template <typename Dtype>
    void tensorrt_gpu_mulbsx(const int num,const Dtype* A,
                           const Dtype* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                           Dtype* B) {
        MulBsx<Dtype> <<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
            num, A, v, rows, cols, trans,
            B);
        CUDA_POST_KERNEL_CHECK;

    }
    template void tensorrt_gpu_divbsx<float>(const int num,const float* A,
                           const float* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                           float* B);
    template void tensorrt_gpu_mulbsx<float>(const int num,const float* A,
                           const float* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
                           float* B);

    // gemv
    template <>
    void tensorrt_gpu_gemv<float>(cublasHandle_t cublas_handle_, const CBLAS_TRANSPOSE TransA, const int M,
                               const int N, const float alpha, const float* A, const float* x,
                               const float beta, float* y) {
        cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_CHECK(cublasSgemv(cublas_handle_, cuTransA, N, M, &alpha,
                                 A, N, x, 1, &beta, y, 1));
    }

    template <>
    void tensorrt_gpu_gemv<double>(cublasHandle_t cublas_handle_, const CBLAS_TRANSPOSE TransA, const int M,
                                const int N, const double alpha, const double* A, const double* x,
                                const double beta, double* y) {
        cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
        CUBLAS_CHECK(cublasDgemv(cublas_handle_, cuTransA, N, M, &alpha,
                                 A, N, x, 1, &beta, y, 1));
    }

    // permute
    template <typename Dtype>
    __global__ void PermuteKernel(const int nthreads, Dtype* const bottom_data, const bool forward,
                       const int* permute_order, const int* old_steps, const int* new_steps,
                       const int num_axes, Dtype* const top_data) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            int temp_idx = index;
            int old_idx = 0;
            for (int i = 0; i < num_axes; ++i) {
                int order = permute_order[i];
                old_idx += (temp_idx / new_steps[i]) * old_steps[order];
                temp_idx %= new_steps[i];
            }
            if (forward) {
                top_data[index] = bottom_data[old_idx];
            } else {
               bottom_data[old_idx] = top_data[index];
            }
        }
    }

    template <typename Dtype>
    void tensorrt_gpu_permute(const int count, Dtype* const bottom_data, const int* permute_order,
                 const int* old_steps, const int* new_steps, const int num_axes, Dtype* const top_data){
        bool forward = true;
        PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, forward, permute_order, old_steps, new_steps,
            num_axes, top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    template void tensorrt_gpu_permute<float>(const int count, float* const bottom_data, const int* permute_order,
                 const int* old_steps, const int* new_steps, const int num_axes, float* const top_data);
}

