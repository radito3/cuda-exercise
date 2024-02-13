#ifndef KERNEL_GPU_H
#define KERNEL_GPU_H

#include "matrix.h"

#ifdef __cplusplus									
extern "C"
#endif // __cplusplus

float gpuMatrixConvulation3D(struct CpuGpuMat* image, struct CpuGpuMat* mask, struct CpuGpuMat* result);

#endif
