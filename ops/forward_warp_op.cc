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

// TODO assert input flow channel count = 2, assert matching numbers in all other dims

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void ForwardWarp(const GPUDevice& d,
                 typename TTypes<float, 4>::ConstTensor input,
                 typename TTypes<float, 4>::Tensor output);

void ForwardWarpGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor input_grad,
                     typename TTypes<float, 4>::ConstTensor original_input,
                     typename TTypes<float, 4>::Tensor output_grad);

class ForwardWarpOp : public OpKernel {
public:
  explicit ForwardWarpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    typename TTypes<float, 4>::ConstTensor input_data = input.tensor<float, 4>();

    const int batch = input_data.dimension(0);
    const int height = input_data.dimension(1);
    const int width = input_data.dimension(2);
    const int channels = input_data.dimension(3);

    auto output_shape = TensorShape({batch, height, width, 1});
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output));
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    ForwardWarp(context->eigen_device<GPUDevice>(),
                input_data, output_data);
  }
};

class ForwardWarpOpGrad : public OpKernel {
public:
  explicit ForwardWarpOpGrad(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& original_input = context->input(1);

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, original_input.shape(),
                                                     &output));

    typename TTypes<float, 4>::ConstTensor input_data = input.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor original_data = original_input.tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    ForwardWarpGrad(context->eigen_device<GPUDevice>(),
                     input_data, original_data, output_data);
  }
};

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

REGISTER_OP("ForwardWarp")
  .Input("flows: float")
  .Output("output: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    ShapeHandle in = c->input(0);
    DimensionHandle batch = c->Dim(in, 0);
    DimensionHandle height = c->Dim(in, 1);
    DimensionHandle width = c->Dim(in, 2);
    c->set_output(0, c->MakeShape({batch, height, width, 1}));
    return Status::OK();
  });

REGISTER_OP("ForwardWarpGrad")
  .Input("grads: float")
  .Input("original_flows: float")
  .Output("output: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(1));
    return Status::OK();
  });

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("ForwardWarp").Device(DEVICE_GPU), ForwardWarpOp);
REGISTER_KERNEL_BUILDER(Name("ForwardWarpGrad").Device(DEVICE_GPU), ForwardWarpOpGrad);

#endif // GOOGLE_CUDA
