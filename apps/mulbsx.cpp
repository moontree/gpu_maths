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

     // mulbsx test
    float *base_data, *mult_vector, *div_vector, *result_vector;
    CUDA_CHECK(cudaMallocManaged(&base_data, c * h * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&result_vector, c * h * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&mult_vector, c * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&div_vector, h * sizeof(float))) ;
    tensorrt_gpu_set<float>(c * h, float(3), base_data);
    tensorrt_gpu_set<float>(c , float(2), mult_vector);
    tensorrt_gpu_set<float>(h , float(4), div_vector);
     /**
     base_data :
         3 3 3 3 3
         3 3 3 3 3
         3 3 3 3 3
     mult_vector:
         2 2 2
     div_vector:
         4 4 4 4 4
     **/
    cout << " base_data * div_vector, noTrans " << endl;
    tensorrt_gpu_mulbsx<float>(c * h, base_data, div_vector, c, h, CblasNoTrans, result_vector);
    cudaMemcpy(local_data, result_vector,c * h * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(local_data, c, h);
    cout << " base_data * mult_vector, Trans " << endl;
    tensorrt_gpu_divbsx<float>(c * h, base_data, mult_vector, c, h, CblasTrans, result_vector);
    cudaMemcpy(local_data, result_vector,c * h * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(local_data, h, c);
    return 0;
                            
}