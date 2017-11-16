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
    memset(local_data, 0 ,sizeof(local_data));
    print_data(local_data, h, w);
    // cuda malloc
    cudaMalloc(&ori_data, h * w * sizeof(float));
    cout << " Test CudaMalloc Pass " << endl;
    CUDA_CHECK(cudaMallocManaged(&ori_data, h * w * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&power_data, h * w * sizeof(float))) ;
    // init
    tensorrt_gpu_set<float>(h * w, float(3), ori_data);
    tensorrt_gpu_powx<float>(h * w, ori_data, float(2), power_data);
    // copy to cpu
    cudaMemcpy(local_data, power_data, h * w * sizeof(float), cudaMemcpyDeviceToHost);
    // output
    print_data(local_data, 5, 5);
    for(int i = 0; i < h * w; i ++){
        assert(local_data[i] == 9);
    }
    cout << "Test powx Pass!" << endl;
    return 0;
                            
}