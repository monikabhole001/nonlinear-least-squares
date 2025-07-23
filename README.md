# GPU-Accelerated Non-Linear Least Squares Solver (OpenCL + C++)

This project implements a parallelized **Non-Linear Least Squares (NLLS)** solver using the **Gauss-Newton algorithm** on the GPU via **OpenCL**. It compares the performance and accuracy of a CPU-based solver versus a GPU-accelerated version, particularly for exponential decay models.

---

## 🚀 Features

- Implements curve fitting for the model:  
  `y = a * exp(-k * t)`
- Solves using the **Gauss-Newton optimization** method
- CPU version uses `Eigen` for matrix algebra
- GPU version uses:
  - OpenCL kernels for residual and Jacobian computation
  - Memory buffers for fast parallel updates
- Benchmarks runtime of CPU vs GPU
- Reports convergence iterations and speedups

---

## 🔧 Technologies Used

- C++
- OpenCL
- Eigen (for CPU-side matrix operations)
- Meson Build System
- KDevelop Project Configuration

---


