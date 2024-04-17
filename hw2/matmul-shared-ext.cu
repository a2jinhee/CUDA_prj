#include <iostream>
#include <cstdio>
#include <stdlib.h> // for rand(), malloc(), free()
#include "common.h"
#include <sys/time.h>

const int TILE_WIDTH    = 32;   // block_dim will be (TILE_WIDTH, TILE_WDITH)

//random data generation 
void genData(float* ptr, unsigned int size) {
    while (size) {
        *ptr++ = (float)size/(float)1000;
        size--;
    }
}

//matmul-shard kernel
__global__ void matmul_shared(float* g_C, const float* g_A, const float* g_B, const int width, const int wh, const int height){

    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y; int bx = blockIdx.x; 
    int ty = threadIdx.y; int tx = threadIdx.x; 
    int gy = by * TILE_WIDTH + ty; // gloabl y index
    int gx = bx * TILE_WIDTH + tx; // global x index
    float sum = 0.0F;
    int iter = (wh + TILE_WIDTH - 1) / TILE_WIDTH;; 

    for (register int m=0; m<iter; ++m){
        // read into the shared memory blocks 
        if (gy < height && (m*TILE_WIDTH + tx) < wh) {
            s_A[ty][tx] = g_A[gy*wh+(m*TILE_WIDTH+tx)]; //flattened array
        } else {
            s_A[ty][tx] = 0.0f; // Padding with zeros for out-of-bounds access
        }
        
        if ((m*TILE_WIDTH + ty) < wh && gx < width) {
            s_B[ty][tx] = g_B[(m*TILE_WIDTH+ty)*width+gx];
        } else {
            s_B[ty][tx] = 0.0f; // Padding with zeros for out-of-bounds access
        }
        __syncthreads(); 
        
        // partial sum for all tiles 
        for (register int k=0; k<TILE_WIDTH; ++k){
            sum += s_A[ty][k] * s_B[k][tx];
        }
        __syncthreads(); 
        if (gy < height && gx < width){
            g_C[gy*width+gx] = sum; 
        }
    }
}

//matmul-global kernel
__global__ void matmul_global(float* g_C, const float* g_A, const float* g_B, const int width, const int wh, const int height){

    int by = blockIdx.y; int bx = blockIdx.x; 
    int ty = threadIdx.y; int tx = threadIdx.x; 
    int gy = by * blockDim.y + ty; // gloabl y index
    int gx = bx * blockDim.x + tx; // global x index
    float sum = 0.0F;

        
    for (register int k=0; k<wh; ++k){
        float lhs = g_A[gy*wh+k];
        float rhs = g_B[k*width+gx];
        sum += lhs*rhs;
    }
    __syncthreads();
    if (gy < height && gx < width){
        g_C[gy*width+gx] = sum; 
    }
}



int main(int argc, char *argv[]){

    // Check if input arguments are provided
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <height> <wh> <width>" << std::endl;
        return -1;
    }

    // Parse input arguments for matrix size
    int height = std::atoi(argv[1]);
    int wh = std::atoi(argv[2]);
    int width = std::atoi(argv[3]);

    //CUDA kernel size setting
    const int GRID_HEIGHT   = (height + TILE_WIDTH - 1) / TILE_WIDTH; // grid_dim will be (GRID_HEIGHT, GRID_WIDTH)
    const int GRID_WIDTH    = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(GRID_WIDTH, GRID_HEIGHT, 1);

    float* pA = NULL; 
    float* pB = NULL; 
    float* pC = NULL; 
    float* pC2 = NULL;
    struct timeval start_time, end_time; 

    // malloc memories on the host-side
    pA = (float*)malloc(height*wh*sizeof(float));
    pB = (float*)malloc(wh*width*sizeof(float));
    pC = (float*)malloc(height*width*sizeof(float));
    pC2 = (float*)malloc(height*width*sizeof(float));
    // generate sourece data
    genData(pA, height*wh);
    genData(pB, wh*width);

    //CUDA: allocate device memory 
    float* pAdev = NULL;
    float* pBdev = NULL;
    float* pCdev = NULL;
    float* pC2dev = NULL;
    CUDA_CHECK( cudaMalloc((void**)&pAdev, height*wh*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&pBdev, wh*width*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&pCdev, height*width*sizeof(float)) );
    CUDA_CHECK( cudaMalloc((void**)&pC2dev, height*width*sizeof(float)) );

    // copy from host to device 
    CUDA_CHECK( cudaMemcpy(pAdev, pA, height*wh*sizeof(float), 
                            cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(pBdev, pB, wh*width*sizeof(float), 
                            cudaMemcpyHostToDevice) );           

    // SHARED 
    // get current time
    cudaThreadSynchronize(); 
    gettimeofday(&start_time, NULL); 

    //CUDA: launch the kernel
    matmul_shared <<<dimGrid, dimBlock>>>(pCdev, pAdev, pBdev, width, wh, height); 
    CUDA_CHECK( cudaPeekAtLastError() ); 

    //get current time
    cudaThreadSynchronize(); 
    gettimeofday(&end_time, NULL); 
    double operating_time = ((double)(end_time.tv_sec)
                                +(double)(end_time.tv_usec)/1000000.0)
                            - ((double)(start_time.tv_sec)
                                +(double)(start_time.tv_usec)/1000000.0);
    printf("Shared Elapsed: %f seconds\n", (double)operating_time);

    // GLOBAL
    // get current time
    cudaThreadSynchronize(); 
    gettimeofday(&start_time, NULL); 

    //CUDA: launch the kernel
    matmul_global <<<dimGrid, dimBlock>>>(pC2dev, pAdev, pBdev, width, wh, height); 
    CUDA_CHECK( cudaPeekAtLastError() ); 

    //get current time
    cudaThreadSynchronize(); 
    gettimeofday(&end_time, NULL); 
    operating_time = ((double)(end_time.tv_sec)
                    +(double)(end_time.tv_usec)/1000000.0)
                - ((double)(start_time.tv_sec)
                    +(double)(start_time.tv_usec)/1000000.0);
    printf("Global Elapsed: %f seconds\n", (double)operating_time);

    // copy from device to host
    CUDA_CHECK( cudaMemcpy(pC, pCdev, width*height*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(pC2, pC2dev, width*height*sizeof(float), cudaMemcpyDeviceToHost) );
    // free device memory
    CUDA_CHECK( cudaFree(pAdev) );
    CUDA_CHECK( cudaFree(pBdev) );
    CUDA_CHECK( cudaFree(pCdev) );
    CUDA_CHECK( cudaFree(pC2dev) );
    
    // print sample cases
    int i, j; 
    i=0; j=0; 
    printf("c[%4d][%4d] = %f\n", i, j, pC[i*width+j]);
    printf("c2[%4d][%4d] = %f\n", i, j, pC2[i*width+j]);
    i = height/2; j=width/2;
    printf("c[%4d][%4d] = %f\n", i, j, pC[i*width+j]);
    printf("c2[%4d][%4d] = %f\n", i, j, pC2[i*width+j]);
    i = height-1; j=width-1;
    printf("c[%4d][%4d] = %f\n", i, j, pC[i*width+j]);
    printf("c2[%4d][%4d] = %f\n", i, j, pC2[i*width+j]);

    // // print the result
    // for (int y=0; y<height; ++y){
    //     for (int x= 0; x<width; ++x){
    //         printf("%.8f ", pC[y*width+x]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // // print the result
    // for (int y=0; y<height; ++y){
    //     for (int x= 0; x<width; ++x){
    //         printf("%.8f ", pC2[y*width+x]);
    //     }
    //     printf("\n");
    // }

    // done 
    return 0; 

}