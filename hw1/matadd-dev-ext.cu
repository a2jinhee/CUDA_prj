#include <iostream>
#include "common.h"


const int TILE_WIDTH    = 32;                   // block_dim will be (TILE_WIDTH, TILE_WDITH)

// Function to perform matrix addition on the CPU
// void addMatrixCPU(int* c, const int* a, const int* b, int width, int height) {
//     for (int y = 0; y < height; ++y) {
//         for (int x = 0; x < width; ++x) {
//             int idx = y * width + x;
//             c[idx] = a[idx] + b[idx];
//         }
//     }
// }

__global__ void addKernel(int* c, const int* a, const int* b, int width, int height){
    // get the 2D position of the thread in the block
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    // check if the thread is within the matrix
    if (x < width && y < height){
        // get the 1D position of the thread in the grid
        int idx = y * width + x; 
        c[idx] = a[idx] + b[idx]; 
    }
}

// main program for the CPU: compiled by MS-VC++
int main(int argc, char *argv[]){
    // Check if input arguments are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <height> <width>" << std::endl;
        return -1;
    }

    // Parse input arguments for matrix size
    int height = std::atoi(argv[1]);
    int width = std::atoi(argv[2]);

    // check maxThreadsPerBlock
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name       : %s\n", prop.name);
    printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    cudaDeviceReset();

    //CUDA kernel size setting
    const int GRID_HEIGHT = (height + TILE_WIDTH - 1) / TILE_WIDTH; // grid_dim will be (GRID_HEIGHT, GRID_WIDTH)
    const int GRID_WIDTH = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    int a[height][width];
    int b[height][width];
    // int c_cpu[height][width] = {0}; // Result matrix for CPU computation
    int c[height][width] = { 0 };

    // make a, b matrices
    for (int y=0; y<height; ++y){
        for (int x=0; x<width; ++x){
            a[y][x] = y*10+x; 
            b[y][x] = (y*10+x)*100; 
        }
    }

    // Perform matrix addition on CPU
    // addMatrixCPU(reinterpret_cast<int*>(c_cpu), reinterpret_cast<const int*>(a), reinterpret_cast<const int*>(b), width, height);

    // device-side data
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    // allocate device memory 
    CUDA_CHECK( cudaMalloc((void**)&dev_a, height*width*sizeof(int)) ); 
    CUDA_CHECK( cudaMalloc((void**)&dev_b, height*width*sizeof(int)) ); 
    CUDA_CHECK( cudaMalloc((void**)&dev_c, height*width*sizeof(int)) ); 
    // copy from host to device
    CUDA_CHECK( cudaMemcpy(dev_a, a, height*width*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dev_b, b, height*width*sizeof(int), cudaMemcpyHostToDevice) );

    // launch a kernel with one thread for each element 
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(GRID_WIDTH, GRID_HEIGHT, 1); 
    addKernel <<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b, width, height); // dev_c = dev_a + dev_b; 
    CUDA_CHECK( cudaPeekAtLastError() ); 

    //copy from device to host
    CUDA_CHECK( cudaMemcpy(c, dev_c, height*width*sizeof(int), cudaMemcpyDeviceToHost) );

    // Compare results
    // bool success = true;
    // for (int y = 0; y < height; ++y) {
    //     for (int x = 0; x < width; ++x) {
    //         if (c_cpu[y][x] != c[y][x]) {
    //             std::cerr << "Error: CPU and GPU results differ at position (" << y << ", " << x << ")" << std::endl;
    //             success = false;
    //             break;
    //         }
    //     }
    //     if (!success) {
    //         break;
    //     }
    // }

    // if (success) {
    //     std::cout << "Results match: CPU and GPU computations are correct." << std::endl;
    // } else {
    //     std::cout << "Results mismatch: CPU and GPU computations are incorrect." << std::endl;
    // }

    // free device memory
    CUDA_CHECK( cudaFree(dev_c) );
    CUDA_CHECK( cudaFree(dev_a) );
    CUDA_CHECK( cudaFree(dev_b) );

    // // print the result
    // for (int y=0; y<height; ++y){
    //     for (int x= 0; x<width; ++x){
    //         printf("%6d", c[y][x]);
    //     }
    //     printf("\n");
    // }

    // done
    return 0; 

}