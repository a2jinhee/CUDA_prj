#include <stdio.h>
#include "support.h"
#define BLOCK_SIZE (TILE_SIZE+FILTER_SIZE-1)

__global__ void convolution(Matrix N, Matrix P)
{
    //INSERT KERNEL CODE HERE
    __shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE]; 
	int ty= threadIdx.y; int tx= threadIdx.x;
	int row_o = blockIdx.y*TILE_SIZE+ ty;
	int col_o = blockIdx.x*TILE_SIZE+ tx; 
	int row_i = row_o - FILTER_SIZE/2; 
	int col_i = col_o - FILTER_SIZE/2; 
	float output = 0.0f;

    if((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width)) { 
        N_ds[ty][tx] = N.elements[row_i * N.width + col_i];
    } else{
        N_ds[ty][tx] = 0.0f;
    }
    __syncthreads();

    if (ty < TILE_SIZE && tx < TILE_SIZE){
        for (int i=0; i<FILTER_SIZE; i++){
            for (int j=0; j<FILTER_SIZE; j++){
                output += Mc[i][j] * N_ds[i+ty][j+tx];
            }
        }
        __syncthreads();
        if (row_o<P.height && col_o<P.width){
            P.elements[row_o*P.width+col_o]=output; 
        }
    }


}
