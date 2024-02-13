#pragma once

struct CpuGpuMat {
	void* cpuP;		// ram pointer
	void* gpuP;		// graphic memory pointer
	int Rows;
	int Cols;
	int Depth;
 
  int Size() {
      return Rows * Cols * Depth;
  }
};
