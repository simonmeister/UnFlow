#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define gauss(x, y, std)

typedef Eigen::GpuDevice GPUDevice;

__global__ void ForwardWarpKernel(const int32 nthreads,
                                  const float* flows,
                                  int batch, int height, int width,
                                  float* output) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = x + width * (y + height * b)
    int idx = out_idx;
    const int src_x = idx % width;
    idx /= width;
    const int src_y = idx % height;
    const int b = idx / height;

    const int flow_index = out_idx * 2;
    const float target_x = src_x + flows[flow_index];
    const float target_y = src_y + flows[flow_index + 1];

    // Calculate distribution variance depending on similar neighbor flows
    // fixed variance for first tests!!

    // Compute valid neighbor range
    //int min_n_y = y + 2 > 0 ? floorf(pos_y) : 0;

    const float dist = 2.0;
    const float std = dist * 0.5;
    const int k = ceilf(dist + 2);
    // TODO variance different for x, y?

    // center pixel closest to mapping location
    //const int closest_x = roundf(target_x);
    //const int closest_y = roundf(target_y);
    if(floorf(target_x - k) < width && floorf(target_x + k) >= 0
        && floorf(target_y - k) < height && floorf(target_y + k) >= 0) {
      const int min_n_x = target_x - k > 0? floorf(target_x - k) : 0;
      const int min_n_y = target_y - k > 0? floorf(target_y - k) : 0;
      const int max_n_x = target_x + k < width? floorf(target_x + k) : width - 1;
      const int max_n_y = target_y + k < height? floorf(target_y + k) : height - 1;

      const float gauss_divisor = 2 * powf(std, 2);
      for(int n_x = min_n_x; n_x <= max_n_x; ++n_x) {
        for(int n_y = min_n_y; n_y <= max_n_y; ++n_y) {
          const float x = n_x - target_x;
          const float y = n_y - target_y;
          const float weight = expf(-(powf(x, 2) + powf(y, 2)) / gauss_divisor);
          CudaAtomicAdd(output + n_x + width * (n_y + height * b), weight);
        }
      }
    }

  }
}

__global__ void ForwardWarpGradKernel(const int32 nthreads,
                                      const float* input_grad, const float* flows,
                                      int batch, int height, int width,
                                      float* output_grad) {
  CUDA_1D_KERNEL_LOOP(in_idx, nthreads) {
    // in_idx =  x + width * (y + height * b)
    int idx = in_idx;
    const int src_x = idx % width;
    idx /= width;
    const int src_y = idx % height;
    const int b = idx / height;

    const int flow_index = in_idx * 2;
    const float target_x = src_x + flows[flow_index];
    const float target_y = src_y + flows[flow_index + 1];

    // Calculate distribution variance depending on similar neighbor flows
    // fixed variance for first tests!!

    // Compute valid neighbor range
    //int min_n_y = y + 2 > 0 ? floorf(pos_y) : 0;

    const float dist = 2.0;
    const float std = dist * 0.5;
    const int k = ceilf(dist + 2);
    // TODO variance different for x, y?

    // center pixel closest to mapping location
    //const int closest_x = roundf(target_x);
    //const int closest_y = roundf(target_y);
    float du = 0.0;
    float dv = 0.0;

    if(floorf(target_x - k) < width && floorf(target_x + k) >= 0
        && floorf(target_y - k) < height && floorf(target_y + k) >= 0) {
      const int min_n_x = target_x - k > 0? floorf(target_x - k) : 0;
      const int min_n_y = target_y - k > 0? floorf(target_y - k) : 0;
      const int max_n_x = target_x + k < width? floorf(target_x + k) : width - 1;
      const int max_n_y = target_y + k < height? floorf(target_y + k) : height - 1;

      const float gauss_divisor = 2 * powf(std, 2);
      for(int n_x = min_n_x; n_x <= max_n_x; ++n_x) {
        for(int n_y = min_n_y; n_y <= max_n_y; ++n_y) {
          const float x = n_x - target_x;
          const float y = n_y - target_y;
          const float weight = expf(-(powf(x, 2) + powf(y, 2)) / gauss_divisor);

          const float din = input_grad[n_x + width * (n_y + height * b)];
          const float factor = 2 * din * weight / gauss_divisor;
          du += factor * x;
          dv += factor * y;
        }
      }
    }

    output_grad[flow_index] = du;
    output_grad[flow_index + 1] = dv;
  }
}

void ForwardWarp(const GPUDevice& d,
                 typename TTypes<float, 4>::ConstTensor flows,
                 typename TTypes<float, 4>::Tensor output) {
  const int batch = flows.dimension(0);
  const int height = flows.dimension(1);
  const int width = flows.dimension(2);

  const int total_count = batch * height * width;
  if (total_count == 0) return;

  CudaLaunchConfig config;

  // Initialize output with all zeros.
  config = GetCudaLaunchConfig(total_count, d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, output.data());

  config = GetCudaLaunchConfig(total_count, d);
  ForwardWarpKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, flows.data(),
      batch, height, width,
      output.data());
}

void ForwardWarpGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor input_grad,
                     typename TTypes<float, 4>::ConstTensor flows,
                     typename TTypes<float, 4>::Tensor output_grad) {
  const int batch = input_grad.dimension(0);
  const int height = input_grad.dimension(1);
  const int width = input_grad.dimension(2);

  int total_count = batch * height * width;
  if (total_count == 0) return;

  // Initialize output_grad with all zeros.
  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, output_grad.data());

  // Accumulate.
  config = GetCudaLaunchConfig(total_count, d);
  ForwardWarpGradKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, input_grad.data(), flows.data(),
      batch, height, width,
      output_grad.data());
}

#endif  // GOOGLE_CUDA
