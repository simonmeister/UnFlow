#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__global__ void BackwardWarpKernel(const int32 nthreads,
                                   const float* images, const float* flows,
                                   int batch, int height, int width, int channels,
                                   float* output) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = src_x + width * (src_y + height * b)
    int idx = out_idx;
    const int src_x = idx % width;
    idx /= width;
    const int src_y = idx % height;
    const int b = idx / height;

    const int flow_index = out_idx * 2;
    const float x = src_x + flows[flow_index];
    const float y = src_y + flows[flow_index + 1];

    const int x0 = floorf(x);
    const int x1 = x0 + 1;
    const int y0 = floorf(y);
    const int y1 = y0 + 1;

    const float w_right = x - x0;
    const float w_left = x1 - x;
    const float w_bottom = y - y0;
    const float w_top = y1 - y;

    for(int c = 0; c < channels; ++c) {
      float sum = 0.0;

#define IMG(iy, ix) images[c + channels * (ix + width * (iy + height * b))]

      // top-left neighbor
	  	if(x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
	  		sum += w_left * w_top * IMG(y0, x0);
	  	}

      // top-right neigbor
	  	if(x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
	  		sum += w_right * w_top * IMG(y0, x1);
	  	}

      // bottom-left neighbor
	  	if(x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
	  		sum += w_left * w_bottom * IMG(y1, x0);
	  	}

      // bottom-right neighbor
	  	if(x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
	  		sum += w_right * w_bottom * IMG(y1, x1);
	  	}
#undef IMG
      output[out_idx * channels + c] = sum;
    }
  }
}

__global__ void BackwardWarpGradKernel(const int32 nthreads,
                                       const float* input_grad,
                                       const float* input_images, const float* flows,
                                       int batch, int height, int width, int channels,
                                       float* output_grad) {
  CUDA_1D_KERNEL_LOOP(in_idx, nthreads) {
    // in_idx =  x + width * (y + height * b)
    int idx = in_idx;
    const int src_x = idx % width;
    idx /= width;
    const int src_y = idx % height;
    const int b = idx / height;

    const int flow_index = in_idx * 2;
    const float x = src_x + flows[flow_index];
    const float y = src_y + flows[flow_index + 1];

    const int x0 = floorf(x);
    const int x1 = x0 + 1;
    const int y0 = floorf(y);
    const int y1 = y0 + 1;

    const float w_right = x - x0;
    const float w_left = x1 - x;
    const float w_bottom = y - y0;
    const float w_top = y1 - y;

    float du = 0.0;
    float dv = 0.0;

    for(int c = 0; c < channels; ++c) {
      float px;
      float din = input_grad[c + channels * in_idx];

#define IMG(iy, ix) input_images[c + channels * (ix + width * (iy + height * b))]

      // top-left neighbor
      if(x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
        px = IMG(y0, x0) * din;
        du -= w_top * px;
        dv -= w_left * px;
      }

      // top-right neigbor
      if(x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
        px = IMG(y0, x1) * din;
        du += w_top * px;
        dv -= w_right * px;
      }

      // bottom-left neighbor
      if(x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
        px = IMG(y1, x0) * din;
        du -= w_bottom * px;
        dv += w_left * px;
      }

      // bottom-right neighbor
      if(x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
        px = IMG(y1, x1) * din;
        du += w_bottom * px;
        dv += w_right * px;
      }
#undef IMG
    }
    output_grad[in_idx * 2] = du;
    output_grad[in_idx * 2 + 1] = dv;
  }
}

void BackwardWarp(const GPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor images,
                  typename TTypes<float, 4>::ConstTensor flows,
                  typename TTypes<float, 4>::Tensor output) {
  const int batch = images.dimension(0);
  const int height = images.dimension(1);
  const int width = images.dimension(2);
  const int channels = images.dimension(3);

  const int total_count = batch * height * width;
  if (total_count == 0) return;

  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  BackwardWarpKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, images.data(), flows.data(),
      batch, height, width, channels,
      output.data());
}

void BackwardWarpGrad(const GPUDevice& d,
                      typename TTypes<float, 4>::ConstTensor input_grad,
                      typename TTypes<float, 4>::ConstTensor input_images,
                      typename TTypes<float, 4>::ConstTensor flows,
                      typename TTypes<float, 4>::Tensor output_grad) {
  const int batch = input_grad.dimension(0);
  const int height = input_grad.dimension(1);
  const int width = input_grad.dimension(2);
  const int channels = input_grad.dimension(3);

  int total_count;
  CudaLaunchConfig config;

  // Initialize output_grad with all zeros.
  total_count = batch * height * width;
  if (total_count == 0) return;
  config = GetCudaLaunchConfig(total_count, d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, output_grad.data());

  // Accumulate.
  total_count = batch * height * width;
  config = GetCudaLaunchConfig(total_count, d);
  BackwardWarpGradKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, input_grad.data(),
      input_images.data(), flows.data(),
      batch, height, width, channels,
      output_grad.data());
}


#endif  // GOOGLE_CUDA
