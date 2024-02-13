#include "device_launch_parameters.h"
#include "matrix.h"
#include "kernel_gpu.cuh"
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>	 // cudaDeviceSynchronize()
#include <iostream>

#ifdef USE_SMALL_IMAGE
#include "small_size.h"
#endif
#ifdef USE_MED_IMAGE
#include "medium_size.h"
#endif
#ifdef USE_LARGE_IMAGE
#include "large_size.h"
#endif

int main() {
  const int maskSize = 3;

  // input
	CpuGpuMat Mat1 = {.Rows = IMAGE_ROWS, .Cols = IMAGE_COLS, .Depth = IMAGE_DEPTH};
  // mask
	CpuGpuMat Mat2 = {.Rows = maskSize, .Cols = maskSize, .Depth = maskSize};
  // result
	CpuGpuMat Mat3 = {.Rows = Mat1.Rows - maskSize + 1, .Cols = Mat1.Cols - maskSize + 1, .Depth = 1};


	// cpu and gpu memory allocation
  Mat1.cpuP = new float[Mat1.Size()];
	Mat2.cpuP = new float[Mat2.Size()]{ 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,	// mean filter
									0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,		// mean filter
									0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370 };	// mean filter
  Mat3.cpuP = new float[Mat3.Size()];

	cudaError_t result1 = cudaMalloc(&Mat1.gpuP, Mat1.Size() * sizeof(float));
	cudaError_t result2 = cudaMalloc(&Mat2.gpuP, Mat2.Size() * sizeof(float));
	cudaError_t result3 = cudaMalloc(&Mat3.gpuP, Mat3.Size() * sizeof(float));
	assert(result1 == cudaSuccess 
        && result2 == cudaSuccess 
        && result3 == cudaSuccess);

	// set values to cpu memory
	for (int i = 0; i < Mat1.Size(); i++) {
    ((float*) Mat1.cpuP)[i] = (float)i;  
  }

	//	Host => ram
	//	Device => graphics memory	

	// Host -> Device
	result1 = cudaMemcpy(Mat1.gpuP, Mat1.cpuP, Mat1.Size() * sizeof(float), cudaMemcpyHostToDevice);
	result2 = cudaMemcpy(Mat2.gpuP, Mat2.cpuP, Mat2.Size() * sizeof(float), cudaMemcpyHostToDevice);
	result3 = cudaMemcpy(Mat3.gpuP, Mat3.cpuP, Mat3.Size() * sizeof(float), cudaMemcpyHostToDevice);
	assert(result1 == cudaSuccess && result2 == cudaSuccess && result3 == cudaSuccess);


  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	// parallel conv
	float constMemCopyTime = gpuMatrixConvulation3D(&Mat1, &Mat2, &Mat3);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float millis = 0;
	cudaEventElapsedTime(&millis, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	// Device -> Host
	cudaError_t result = cudaMemcpy(Mat3.cpuP, Mat3.gpuP, Mat3.Size() * sizeof(float), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);

	cudaDeviceSynchronize();

  #ifdef USE_SHARED_MEM
  std::cout << "Using shared memory for input image and constant memory for mask(kernel)" << std::endl;
  millis -= constMemCopyTime;
  #endif
  std::cout << "Input Size: (rows: " << IMAGE_ROWS << ", cols: "
            << IMAGE_COLS << ", depth: " << IMAGE_DEPTH << ")" 
            << std::endl << "Elapsed time: " << millis << " ms" << std::endl;

	// show result
	for (size_t row = 0; row < Mat3.Rows; row++) {
		for (size_t col = 0; col < Mat3.Cols; col++) {
			std::cout << ((float*) Mat3.cpuP)[row * Mat3.Cols + col] << " ";
		}
		std::cout << std::endl;
	}
	

	// cpu and gpu memory free
	cudaFree(Mat1.gpuP);
	cudaFree(Mat2.gpuP);
	cudaFree(Mat3.gpuP);

	delete[] (float*) Mat1.cpuP;
	delete[] (float*) Mat2.cpuP;
	delete[] (float*) Mat3.cpuP;

	return 0;
}
