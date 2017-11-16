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
    float *ori_data, *power_data;
    float local_data[75];
    // cublas test
    //tensorrt_gpu_asum<float>(cublasHandle_t cublas_handle_, const int n, const float* x, float* y)
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
    // asum test
    CUDA_CHECK(cudaMallocManaged(&ori_data, h * w * sizeof(float))) ;
    // init
    tensorrt_gpu_set<float>(h * w, float(3), ori_data);
    float *sum , local_sum;
    CUDA_CHECK(cudaMallocManaged(&sum,sizeof(float))) ;
    cout << "test1" << local_sum << endl;
    tensorrt_gpu_asum<float>(handle_, h * w, ori_data, sum);
    cout << "test2 " << local_sum << endl;
    cudaMemcpy(&local_sum, sum, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "sum of ori_data is " << local_sum << endl;
    

    return 0;
                            
}