#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

__global__ void DownsampleKernel(const int32 nthreads,
                                 const float* images,
                                 int batch, int in_height, int in_width, int channels,
                                 int out_height, int out_width,
                                 float* output) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = x + out_width * (y + out_height * b)
    int idx = out_idx;
    const int c = idx % channels;
    idx /= channels;
    const int x = idx % out_width;
    idx /= out_width;
    const int y = idx % out_height;
    const int b = idx / out_height;

    const int scale_y = in_height / out_height;
    const int scale_x = in_width/ out_width;

    const int min_in_y = y * scale_y;
    const int min_in_x = x * scale_x;
    const int max_in_y = min_in_y + scale_y;
    const int max_in_x = min_in_x + scale_x;

    float sum = 0.0;

    for(int in_y = min_in_y; in_y < max_in_y; ++in_y) {
      for(int in_x = min_in_x; in_x < max_in_x; ++in_x) {
        sum += images[c + channels * (in_x + in_width * (in_y + in_height * b))];
      }
    }

    sum /= scale_x * scale_y;
    output[c + channels * (x + out_width * (y + out_height * b))] = sum;
  }
}

void Downsample(const GPUDevice& d,
                typename TTypes<float, 4>::ConstTensor images,
                typename TTypes<float, 4>::Tensor output) {
  const int batch = images.dimension(0);
  const int in_height = images.dimension(1);
  const int in_width = images.dimension(2);
  const int channels = images.dimension(3);

  const int out_height = output.dimension(1);
  const int out_width = output.dimension(2);

  const int total_count = batch * out_height * out_width * channels;
  if (total_count == 0) return;

  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  DownsampleKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      config.virtual_thread_count, images.data(),
      batch, in_height, in_width, channels,
      out_height, out_width,
      output.data());
}

#endif  // GOOGLE_CUDA
