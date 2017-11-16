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
    printf("Normalize Layer Enqueue \n");
    // float kONE = 1.0f, kZERO = 0.0f;
    cublasHandle_t mCublas;
    cublasStatus_t status = cublasCreate(&mCublas);  
    float eps = 1e-10;
    // init
    float *norm_data, *sum_channel_multiplier_data, *scale_data, *buffer_data;
    CUDA_CHECK(cudaMallocManaged(&buffer_data, c * h * w * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&norm_data, h * w * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&sum_channel_multiplier_data, c * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&scale_data, c * sizeof(float))) ;
    tensorrt_gpu_set<float>(h * w, eps, norm_data);
    tensorrt_gpu_set<float>(c, float(1.0), sum_channel_multiplier_data);
    tensorrt_gpu_set<float>(c, float(20.0), scale_data);
    float *input, *output;
    for(int i = 0; i < 75; i ++){
        local_data[i] = int(i / 25) + 1;
    }
    print_data(local_data, c * h, w);
    CUDA_CHECK(cudaMallocManaged(&input,c * h * w * sizeof(float))) ;
    CUDA_CHECK(cudaMallocManaged(&output,c * h * w * sizeof(float))) ;
    cudaMemcpy(input, local_data, c * h * w *  sizeof(int), cudaMemcpyHostToDevice);
    // data init
    const float *bottom_data = reinterpret_cast<const float*>(input);
    float *top_data = reinterpret_cast<float*>(output);
    const float *scale = reinterpret_cast<const float*>(scale_data);

    int num = 1;
    int dim = c * h * w;
    int spatial_dim = h * w;
    int channels = c;
    bool across_spatial_ = false;
    bool channel_shared_ = false;
    printf("num is %d\n",num);
    printf("dim is %d\n",dim);
    printf("spatial_dim is %d\n",spatial_dim);
    printf("channels is %d\n",channels);
    for(int n = 0; n < num; n++){
        // buffer_data[i] = bottom_data[i] ^ 2
        tensorrt_gpu_powx<float>(dim, bottom_data, float(2), buffer_data);
        if(across_spatial_){
            float normsqr;
            tensorrt_gpu_asum<float>(mCublas, dim, buffer_data, &normsqr);
            norm_data[n] = pow(normsqr + eps, float(0.5));
            // top_data = bottom_data / norm_data[n]
            tensorrt_gpu_scale<float>(mCublas, dim, float(1.0 / norm_data[n]), bottom_data, top_data);
        }else{
            tensorrt_gpu_gemv<float>(mCublas, CblasTrans, channels, spatial_dim, float(1),
                                     buffer_data, sum_channel_multiplier_data, float(1),
                                     norm_data);            
            tensorrt_gpu_powx<float>(spatial_dim, norm_data, float(0.5), norm_data);
           
            tensorrt_gpu_divbsx<float>(dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans, top_data);            
            norm_data += spatial_dim;
           
        }
      
        // scale the output
        if(channel_shared_){
            tensorrt_gpu_scale<float>(mCublas, dim, scale[0], bottom_data, top_data);
        }else{
            tensorrt_gpu_mulbsx<float>(dim, top_data, scale, channels, spatial_dim, CblasTrans, top_data);            
            
        }
        bottom_data += dim;
        top_data += dim;
    }
    cudaMemcpy(local_data, output, c * h * w * sizeof(float), cudaMemcpyDeviceToHost);
    print_data(local_data, c * h, w);
    return 0;
                            
}