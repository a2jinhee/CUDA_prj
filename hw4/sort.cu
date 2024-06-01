#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define MAX_VALUE 100

// allowed? 
#define BLOCK_SIZE 256

// INSERT CODE HERE---------------------------------
//
// CPU SORT FUNCTION
void sort(int* Source, int* Result_CPU, long long input_size)
{
    int count[MAX_VALUE + 1] = {0};
    
    // Count the occurrences of each value in Source
    for (long long i = 0; i < input_size; i++) {
        count[Source[i]]++;
    }
    
    // Compute the prefix sum (scan) of counts
    for (int i = 1; i <= MAX_VALUE; i++) {
        count[i] += count[i - 1];
    }
    
    // Place elements in Result_CPU based on the prefix sum
    for (long long i = input_size - 1; i >= 0; i--) {
        Result_CPU[--count[Source[i]]] = Source[i];
    }
}

// GPU kernel for counting sort with scan (prefix sum)
// GPU kernel for creating the count array
__global__ void gpu_makecount(int* Source, int* Count, long long input_size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // Make count array using Source
    if (gid < input_size) {
        atomicAdd(&Count[Source[gid]], 1);
    }
}

// GPU kernel for performing the prefix sum (scan) on the count array
__global__ void gpuscan_prefix(int* Count)
{
    __shared__ int count[MAX_VALUE + 1];
    int tid = threadIdx.x;

    // Copy global Count to shared memory
    if (tid < (MAX_VALUE + 1)) {
        count[tid] = Count[tid];
    }
    __syncthreads();

    // Brent-Kung scan
    // Reduction step 
    int stride = 1; 
    while (stride < BLOCK_SIZE) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE) {
            count[index] += count[index - stride];
        }
        stride *= 2;
        __syncthreads(); 
    }
    // Scan step 
    int stride2 = BLOCK_SIZE / 2;
    while (stride2 > 0) {
        int index = (tid + 1) * stride2 * 2 - 1;
        if (index < BLOCK_SIZE && (index + stride2) < BLOCK_SIZE) {
            count[index + stride2] += count[index];
        }
        stride2 /= 2;
        __syncthreads();
    }


    // Copy scanned counts from shared memory to global memory
    if (tid < (MAX_VALUE + 1)) {
        Count[tid] = count[tid];
    }
}

// GPU kernel for sorting based on the scanned counts
__global__ void gpusort(int* Source, int* Result_GPU, int* Count, long long input_size)
{
    int gid = threadIdx.x + blockIdx.x * blockDim.x;

    // Place elements in Result_GPU based on the scanned counts
    if (gid < input_size) {
        int pos = atomicSub(&Count[Source[gid]], 1) - 1;
        Result_GPU[pos] = Source[gid];
    }
}
// INSERT CODE HERE---------------------------------


void verify(int* result_cpu, int* result_gpu, long long input_size){
	printf("Verifying results...\n"); 
	fflush(stdout);

	long long match_cnt = 0;
	for(int i = 0; i < input_size; i++)
	{
		if(result_cpu[i] == result_gpu[i]) {
			match_cnt++;
		}
			
	}

	if(match_cnt == input_size)
		printf("TEST PASSED\n\n");
	else
		printf("TEST FAILED\n\n");

}

void genData(int* ptr, long long size) {
	while (size--) {
		*ptr++ = (int)(rand() % MAX_VALUE + 1);
	}
}

int main(int argc, char* argv[]) {
	int* Source = NULL;
	int* Result_CPU = NULL;
	int* Result_GPU = NULL;
	long long input_size = 0;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (argc == 2)
		input_size = (long long)atoi(argv[1]);
	else
	{
    		printf("\n    Invalid input parameters!"
					"\n    Usage: ./sort <input_size>"
					"\n");
        	exit(0);
	}

	// allocate host memory
	Source     = (int*)malloc(input_size*sizeof(int));
	Result_CPU = (int*)malloc(input_size*sizeof(int));
	Result_GPU = (int*)malloc(input_size*sizeof(int));
	// generate source data
	genData(Source, input_size);
	
	
	// start timer
	cudaEventRecord(start, 0);


	// INSERT CODE HERE------------------------------
	//
	//   DEVICE MEMORY ALLOCATION
	int* Source_d = NULL;
    int* Result_GPU_d = NULL;
    // int Count[MAX_VALUE + 1] = {0};
    int* Count_d = NULL;
    cudaMalloc((void**)&Source_d, input_size*sizeof(int));
    cudaMalloc((void**)&Result_GPU_d, input_size*sizeof(int));
    cudaMalloc((void**)&Count_d, (MAX_VALUE + 1)*sizeof(int));
    
    // MEMORY COPY
    cudaMemcpy(Source_d, Source, input_size*sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(Count_d, Count, (MAX_VALUE + 1)*sizeof(int), cudaMemcpyHostToDevice);

    // KERNEL LAUNCH
    dim3 dimGrid((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE);
    gpu_makecount<<<dimGrid, dimBlock>>>(Source_d, Count_d, input_size);
    cudaDeviceSynchronize();

    gpuscan_prefix<<<1, BLOCK_SIZE>>>(Count_d);
    cudaDeviceSynchronize();

    gpusort<<<dimGrid, dimBlock>>>(Source_d, Result_GPU_d, Count_d, input_size);
    cudaDeviceSynchronize();

    cudaMemcpy(Result_GPU, Result_GPU_d, input_size*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(Count, Count_d, (MAX_VALUE + 1)*sizeof(int), cudaMemcpyDeviceToHost);

    // end timer
	//
	// INSERT CODE HERE------------------------------


	// end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("elapsed time = %f msec\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	
	sort(Source, Result_CPU, input_size);

	// allowed?
	// print Count
	// printf("Count: ");
	// for (int i = 0; i < MAX_VALUE + 1; i++)
	// {
	// 	printf("%d ", Count[i]);
	// }
	// // print Source
	// printf("\nSRC: ");
	// for(int i = 0; i < input_size; i++)
	// {
	// 	printf("%d ", Source[i]);
	// }
	// // print Result_CPU
	// printf("\nCPU: ");
	// for(int i = 0; i < input_size; i++)
	// {
	// 	printf("%d ", Result_CPU[i]);
	// }
	// // print Result_GPU
	// printf("\nGPU: ");
	// for(int i = 0; i < input_size; i++)
	// {
	// 	printf("%d ", Result_GPU[i]);
	// }
	// printf("\n");


	verify(Result_CPU, Result_GPU, input_size);
	fflush(stdout);
	
	
	// INSERT CODE HERE--------------------
	//
	//   FREE ALLOCATED MEMORY
	cudaFree(Source_d);
    cudaFree(Result_GPU_d);
    cudaFree(Count_d);
	//
	// INSERT CODE HERE--------------------
}

