//////////////////////////////////////////////////////////////////////////////
// OpenCL Non-Linear Least Square
//////////////////////////////////////////////////////////////////////////////

// includes
#include <stdio.h>
#include <Core/Assert.hpp>
#include <Core/Image.hpp>
#include <Core/Time.hpp>
#include <OpenCL/Device.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/cl-patched.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>
// Eigen library headers for matrix and vector operations
#include <Eigen/Dense>
using namespace Eigen;


// Model function y = a * exp(-k * t) for CPU implementation
double modelFunction_cpu(double t_cpu, double a_cpu, double k_cpu) {
    return a_cpu * exp(-k_cpu * t_cpu);
}

// Partial derivative of the model with respect to a
double partialDerivativeA_cpu(double t_cpu, double k_cpu) {
    return exp(-k_cpu * t_cpu);
}

// Partial derivative of the model with respect to k
double partialDerivativeK_cpu(double t_cpu, double a_cpu, double k_cpu) {
    return -t_cpu * a_cpu * exp(-k_cpu * t_cpu);
}

void gpu(Core::TimeSpan cpuTime){
  try {

  // Initial guesses for parameters a and k
  double a_initial = 200.0;
  double k_initial = 0.1;
  // Create a context
  cl::Context context(CL_DEVICE_TYPE_GPU);
  // Get the first device of the context
  std::cout << "Context has " << context.getInfo<CL_CONTEXT_DEVICES>().size()
            << " devices" << std::endl;
  cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];

  std::vector<cl::Device> devices;
  devices.push_back(device);
  OpenCL::printDeviceInfo(std::cout, device);
  // Create a command queue
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);
  // Load the source code
  extern unsigned char NonLinearLeastSquare_cl[];
  extern unsigned int NonLinearLeastSquare_cl_len;

  cl::Program program(context,std::string((const char*)NonLinearLeastSquare_cl, NonLinearLeastSquare_cl_len));
  // Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
  OpenCL::buildProgram(program, devices);
  // Create a kernel object
  cl::Kernel kernel1(program, "kernel1");

  // Gauss-Newton algorithm data
  std::vector<double> x = {0, 20, 40, 60, 80, 100, 120, 140};
  std::vector<double> y = {147.8, 78.3, 44.7, 29.5, 15.2, 7.8, 3.2, 3.9};
  std::size_t dataSize = x.size(); // Number of data points
  int maxIterations = 100; // Maximum number of iterations for the Gauss-Newton algorithm
  double tolerance = 1e-6;

  // Buffers
        cl::Buffer xBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize * sizeof(double), x.data());
        // When creating buffers, no event is needed as CL_MEM_COPY_HOST_PTR is synchronous
        cl::Buffer yBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dataSize * sizeof(double), y.data());
        cl::Buffer J0Buffer(context, CL_MEM_READ_WRITE, dataSize * sizeof(double));
        cl::Buffer J1Buffer(context, CL_MEM_READ_WRITE, dataSize * sizeof(double));
        cl::Buffer rBuffer(context, CL_MEM_READ_WRITE, dataSize * sizeof(double));

  // Work items configuration
        std::size_t wgSize = 1;  // Number of work items per work group
        std::size_t count = 8;   // Overall number of work items = Number of elements
  // Set the kernel arguments
        kernel1.setArg(0, xBuffer);
        kernel1.setArg(1, yBuffer);
        kernel1.setArg(2, J0Buffer);
        kernel1.setArg(3, J1Buffer);
        kernel1.setArg(4, rBuffer);
        kernel1.setArg(5, a_initial); // Initial guess for a
        kernel1.setArg(6, k_initial);   // Initial guess for k
        kernel1.setArg(7, static_cast<int>(dataSize));

        cl::Event kernelExecution;
        cl::Event copy1;
        cl::Event copy2;
        cl::Event copy3;
        cl::NDRange global(count);
        cl::NDRange local(wgSize);

for (int iter = 0; iter < maxIterations; ++iter) {
        // Execute the kernel
        queue.enqueueNDRangeKernel(kernel1, cl::NullRange, global, local, NULL, &kernelExecution);
        // Read back results
        std::vector<double> J0(dataSize), J1(dataSize), r(dataSize);
        queue.enqueueReadBuffer(J0Buffer, CL_TRUE, 0, dataSize * sizeof(double), J0.data(),NULL, &copy1);
        queue.enqueueReadBuffer(J1Buffer, CL_TRUE, 0, dataSize * sizeof(double), J1.data(),NULL, &copy2);
        queue.enqueueReadBuffer(rBuffer, CL_TRUE, 0, dataSize * sizeof(double), r.data(),NULL, &copy3);

        MatrixXd J(dataSize, 2);
        VectorXd rVec(dataSize);
        for (size_t i = 0; i < dataSize; ++i) {
            J(i, 0) = J0[i]; // J0 is filled with partial derivatives w.r.t a
            J(i, 1) = J1[i]; // J1 is filled with partial derivatives w.r.t k
            rVec(i) = r[i];  // Residuals
        }

         //std::cout << "Jacobian matrix J:\n" << J << "\n\n";
         //std::cout << "RVector :\n" << rVec << "\n\n";

        // Compute Gauss-Newton parameter update
        VectorXd deltaY = (J.transpose() * J).ldlt().solve(J.transpose() * rVec);

        // Update parameters
        a_initial = a_initial + deltaY(0);
        k_initial = k_initial + deltaY(1);
        // Update kernel arguments for current iteration
        kernel1.setArg(5, a_initial);
        kernel1.setArg(6, k_initial);
        //std::cout << "Updated parameters gpu: a = " << a_initial << ", k = " << k_initial << std::endl;
        // Check for convergence (if the magnitude of parameter updates is small enough)
        if (deltaY.norm() < tolerance) {
        std::cout << "GPU Converged in " << iter + 1 << " iterations." << std::endl;
        break;
        }
    }
    std::cout << "Updated parameters gpu: a = " << a_initial << ", k = " << k_initial << std::endl;
    Core::TimeSpan gpuTime = OpenCL::getElapsedTime(kernelExecution);
    // Measure copy times
    Core::TimeSpan copyTime1 = OpenCL::getElapsedTime(copy1);
    Core::TimeSpan copyTime2 = OpenCL::getElapsedTime(copy2);
    Core::TimeSpan copyTime3 = OpenCL::getElapsedTime(copy3);
    Core::TimeSpan copyTime = copyTime1 + copyTime2;
    Core::TimeSpan overallGpuTime = gpuTime + copyTime;
    std::cout << "GPU Time " << overallGpuTime.toString()<< std::endl;
    std::cout << "Updated parameters gpu: a_gpu = " << a_initial << ", k_gpu = " << k_initial << std::endl;
    std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
    std::cout << "GPU Time w/o memory copy: " << gpuTime.toString()<< " (speedup = " << (cpuTime.getSeconds() / gpuTime.getSeconds())<< ")" << std::endl;
    std::cout << "GPU Time with memory copy: " << overallGpuTime.toString()<< " (speedup = "
            << (cpuTime.getSeconds() / overallGpuTime.getSeconds()) << ")"<< std::endl;
    }
     // Handle errors
      catch (const OpenCL::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << ", " << e.err() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////
void cpu(){

    std::vector<double> x_cpu = {0, 20, 40, 60, 80, 100, 120, 140};
    std::vector<double> y_cpu = {147.8, 78.3, 44.7, 29.5, 15.2, 7.8, 3.2, 3.9};

    // Initial guesses for parameters a and k
    double a_cpu = 200;
    double k_cpu = 0.1;

    int maxIterations_cpu = 1000; // Maximum number of iterations
    double tolerance_cpu = 1e-6; // Convergence tolerance
    int n_cpu = x_cpu.size(); // Number of data points

    for(int iter_cpu = 0; iter_cpu < maxIterations_cpu; ++iter_cpu) {
        Eigen::MatrixXd J_cpu(n_cpu, 2); // Jacobian matrix
        Eigen::VectorXd r_cpu(n_cpu); // Residual vector
        Eigen::VectorXd deltaY_cpu(2); // Delta parameters vector
        // Construct the Jacobian matrix and the residual vector
        for(int i_cpu = 0; i_cpu < n_cpu; ++i_cpu) {
            double ti_cpu = x_cpu[i_cpu];
            double yi_cpu = y_cpu[i_cpu];
            double fi_cpu = modelFunction_cpu(ti_cpu, a_cpu, k_cpu);
            J_cpu(i_cpu, 0) = partialDerivativeA_cpu(ti_cpu, k_cpu);
            J_cpu(i_cpu, 1) = partialDerivativeK_cpu(ti_cpu, a_cpu, k_cpu);
            r_cpu(i_cpu) = yi_cpu - fi_cpu;
        }
        deltaY_cpu = (J_cpu.transpose() * J_cpu).ldlt().solve(J_cpu.transpose() * r_cpu);
        // Update the parameters
        a_cpu += deltaY_cpu(0);
        k_cpu += deltaY_cpu(1);
        //std::cout << "a_cpu = " << a_cpu << ", k_cpu = " << k_cpu << std::endl;
        // Check for convergence
        if(deltaY_cpu.norm() < tolerance_cpu) {
            std::cout << "CPU Converged in " << iter_cpu + 1 << " iterations." << std::endl;
            break;
        }
    }
    std::cout << "Estimated parameters:" << std::endl;
    std::cout << "a_cpu = " << a_cpu << ", k_cpu = " << k_cpu << std::endl;
}

int main(int argc, char** argv){
      // Time calculation on the host side
      Core::TimeSpan cpuStart = Core::getCurrentTime();
      cpu();
      Core::TimeSpan cpuEnd = Core::getCurrentTime();
      // Print performance data
      Core::TimeSpan cpuTime = cpuEnd - cpuStart;
      std::cout << "CPU Time: " << cpuTime.toString() << std::endl;
      gpu(cpuTime);
      return 0;
}
