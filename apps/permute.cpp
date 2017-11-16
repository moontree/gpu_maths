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
    // permute test
    int old_steps[4] = {75, 25, 5, 1};
    int new_steps[4] = {75, 15, 3, 1};
    int permute_order[4] = {0, 2, 3, 1};
    int num_axes = 4;
    float *permute_data, *bottom_data;
    int *gpu_old_steps, *gpu_new_steps, *gpu_order;
    CUDA_CHECK(cudaMallocManaged(&bottom_data, n * c * h * w * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&permute_data, n * c * h * w * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&gpu_old_steps,4 * sizeof(int))) ;
    CUDA_CHECK(cudaMallocManaged(&gpu_new_steps,4 * sizeof(int))) ;
    CUDA_CHECK(cudaMallocManaged(&gpu_order,4 * sizeof(int))) ;
    cudaMemcpy(gpu_old_steps, old_steps, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_new_steps, new_steps, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_order, permute_order, 4 * sizeof(int), cudaMemcpyHostToDevice);
    tensorrt_gpu_set<float>(h * w, float(1), bottom_data);
    tensorrt_gpu_set<float>(h * w, float(2), bottom_data + 25);
    tensorrt_gpu_set<float>(h * w, float(3), bottom_data + 50);
    tensorrt_gpu_permute(n * c * h * w, bottom_data, gpu_order, gpu_old_steps, gpu_new_steps, num_axes, permute_data);
    cudaMemcpy(local_data, permute_data,n * c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(local_data, h * w, c);
    return 0;
                            
}