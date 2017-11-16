#include "math_util.hpp"
#include <iostream>
#include <cublas_v2.h>
#include <cblas.h>
using namespace std;
using namespace tensorrt;

void print_data(float* data, int h, int w){
    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            cout << data[i * w + j] << " ";
        }
        cout << endl;
    }
}

int main(){
    int n = 1, c = 3, h = 5, w = 5;
    float local_data[75];
    cublasHandle_t handle_;
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED)
        {
            cout << "Fail to get an instance of blas object! Check whether you have free the handle!" << endl;
        }
        getchar();
        return EXIT_FAILURE;
    }
    // gemv test 
    float *base_data, *mult_vector, *div_vector;
    CUDA_CHECK(cudaMallocManaged(&base_data, c * h * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&mult_vector, c * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&div_vector, h * sizeof(float))) ;
    tensorrt_gpu_set<float>(c * h, float(3), base_data);
    tensorrt_gpu_set<float>(c , float(2), mult_vector);
    tensorrt_gpu_set<float>(h , float(4), div_vector);
    // gemv, no trans
    cout << "gemv test, notransA" << endl;
    tensorrt_gpu_gemv<float>(handle_, CblasNoTrans, c, h, float(1), base_data, div_vector, float(0), mult_vector);
    cudaMemcpy(local_data, mult_vector,c * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(local_data, 1, c);
    tensorrt_gpu_set<float>(c * h, float(3), base_data);
    tensorrt_gpu_set<float>(c , float(2), mult_vector);
    tensorrt_gpu_set<float>(h , float(4), div_vector);
    // gemv, transA
    cout << "gemv test, transA" << endl;
    tensorrt_gpu_gemv<float>(handle_, CblasTrans, c, h, float(1), base_data, mult_vector, float(1), div_vector);
    cudaMemcpy(local_data, div_vector,h * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(local_data, 1, h);
    

    return 0;
                            
}