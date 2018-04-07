#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

void Downsample(const GPUDevice& d,
                typename TTypes<float, 4>::ConstTensor images,
                typename TTypes<float, 4>::Tensor output);

class DownsampleOp : public OpKernel {
public:
  explicit DownsampleOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("scale", &scale));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    typename TTypes<float, 4>::ConstTensor input_data = input.tensor<float, 4>();


    OP_REQUIRES(context,
                input_data.dimension(1) % scale == 0 &&
                  input_data.dimension(2) % scale == 0,
                errors::InvalidArgument("Input height and width must be divisible by scale"));

    const int batch = input_data.dimension(0);
    const int height = input_data.dimension(1) / scale;
    const int width = input_data.dimension(2) / scale;
    const int channels = input_data.dimension(3);

    auto output_shape = TensorShape({batch, height, width, channels});

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    Downsample(context->eigen_device<GPUDevice>(), input_data, output_data);
  }
private:
  int scale;
};

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("Downsample")
  .Input("images: float")
  .Attr("scale: int = 2")
  .Output("out_images: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    ShapeHandle in = c->input(0);
    int scale;
    DimensionHandle batch = c->Dim(in, 0);
    DimensionHandle channels = c->Dim(in, 3);
    DimensionHandle height;
    DimensionHandle width;

    c->GetAttr("scale", &scale);
    c->Divide(c->Dim(in, 1), scale, true, &height);
    c->Divide(c->Dim(in, 2), scale, true, &width);

    c->set_output(0, c->MakeShape({batch, height, width, channels}));
    return Status::OK();
  });

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("Downsample").Device(DEVICE_GPU), DownsampleOp);

#endif // GOOGLE_CUDA
