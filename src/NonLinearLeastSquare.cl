#ifndef __OPENCL_VERSION__
#include "../lib/OpenCL/OpenCLKernel.hpp"
#endif

__kernel void kernel1(__global const double* x,__global const double* y,__global double* J0,
                                   __global double* J1, __global double* r, const double a, const double k,const int dataSize) {
      int i = get_global_id(0);  // Get the unique ID
      if (i < dataSize) {
            double predicted = a * exp(-k * x[i]);
            r[i] = y[i] - predicted; // Calculate the residual
            J0[i] = exp(-k * x[i]); // Partial derivative with respect to a
            J1[i] = -a * x[i] * exp(-k * x[i]);// Partial derivative with respect to k
      }
    }
