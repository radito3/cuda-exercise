#include "device_launch_parameters.h"
#include "matrix.h"
#include "kernel_gpu.cuh"
#include <math.h>
#include <stdio.h>

#ifdef USE_SMALL_IMAGE
#include "small_size.h"
#endif
#ifdef USE_MED_IMAGE
#include "medium_size.h"
#endif
#ifdef USE_LARGE_IMAGE
#include "large_size.h"
#endif

const int maskSize = 3;
const int threadsPerBlock = 32;
const int MAX_MASK_SIZE = maskSize * maskSize * maskSize;
const int BLOCK_SIZE_X = threadsPerBlock;
const int BLOCK_SIZE_Y = threadsPerBlock;

__constant__ float d_mask[MAX_MASK_SIZE];

__global__ void gpuMatrixConv3D(float* image, float* mask, float* result, int imageRows, int imageCols, 
                                int resultRows, int resultCols) { 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0;

	if (row < resultRows && col < resultCols) {
		int imageRowsCols = IMAGE_ROWS * IMAGE_COLS;

		for (int maskRow = 0; maskRow < maskSize; maskRow++) {
			for (int maskCol = 0; maskCol < maskSize; maskCol++) {
				for (int dep = 0; dep < maskSize; dep++) {
					// accessing an element from a flattened 3D structure is with [x(currentCol) * Rows * Depth + y(currentRow) * Depth + z(currentDepth)]
					sum += image[(row + maskRow) * IMAGE_COLS + col + maskCol + dep * imageRowsCols] * mask[maskRow * maskSize + maskCol + dep * maskSize];
				}
			}
		}
		result[row * resultCols + col] = sum;
	}
}

__global__ void gpuMatrixConv3DEnhanced(float* image, float* result, int imageRows, int imageCols, int maskRC,
										int maskDepth, int resultRows, int resultCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    if (row < resultRows && col < resultCols) {
        __shared__ float sharedImage[BLOCK_SIZE_Y + MAX_MASK_SIZE - 1][BLOCK_SIZE_X + MAX_MASK_SIZE - 1];

        // Load data into shared memory
        for (int dep = 0; dep < maskDepth; dep++) {
            sharedImage[threadIdx.y + dep][threadIdx.x] = image[(row + dep) * imageCols + col];
        }

        __syncthreads();

        // Compute convolution using shared memory
        for (int maskRow = 0; maskRow < maskRC; maskRow++) {
            for (int maskCol = 0; maskCol < maskRC; maskCol++) {
                for (int dep = 0; dep < maskDepth; dep++) {
                	sum += sharedImage[threadIdx.y + dep][threadIdx.x + maskCol] * d_mask[maskRow * maskRC + maskCol + dep * maskDepth];
                }
            }
        }

        result[row * resultCols + col] = sum;
    }
}

float gpuMatrixConvulation3D(struct CpuGpuMat* image, struct CpuGpuMat* mask, struct CpuGpuMat* result) {
	#ifdef USE_SHARED_MEM

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	cudaMemcpyToSymbol(d_mask, mask->cpuP, mask->Rows * mask->Cols * mask->Depth * sizeof(float));

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float millis = 0;
	cudaEventElapsedTime(&millis, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("memory copy time: %.6f ms\n", millis);

	int gridCols = ceil(float(result->Cols) / float(threadsPerBlock));
	int gridRows = ceil(float(result->Rows) / float(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	gpuMatrixConv3DEnhanced<<<gridDim, blockDim>>>((float*)image->gpuP, (float*)result->gpuP, image->Rows, image->Cols, mask->Rows, mask->Depth, result->Rows, result->Cols);
	return millis;
	
	#else
	
	int gridCols = ceil(float(result->Cols) / float(threadsPerBlock));
	int gridRows = ceil(float(result->Rows) / float(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock); // total 32*32 = 1024 threads

	gpuMatrixConv3D<<<gridDim, blockDim>>>((float*)image->gpuP, (float*)mask->gpuP, (float*)result->gpuP,
	  									   image->Rows, image->Cols, result->Rows, result->Cols);
	return 0;
	#endif
}
