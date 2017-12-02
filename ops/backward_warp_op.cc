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

typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void BackwardWarp(const GPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor images,
                  typename TTypes<float, 4>::ConstTensor flows,
                  typename TTypes<float, 4>::Tensor output);

void BackwardWarpGrad(const GPUDevice& d,
                      typename TTypes<float, 4>::ConstTensor input_grad,
                      typename TTypes<float, 4>::ConstTensor input_images,
                      typename TTypes<float, 4>::ConstTensor flows,
                      typename TTypes<float, 4>::Tensor output_grad);

class BackwardWarpOp : public OpKernel {
public:
  explicit BackwardWarpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_images = context->input(0);
    const Tensor& input_flows = context->input(1);

    Tensor* output_images = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_images.shape(),
                                                     &output_images));

    typename TTypes<float, 4>::ConstTensor image_data = input_images.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor flow_data = input_flows.tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_data = output_images->tensor<float, 4>();

    BackwardWarp(context->eigen_device<GPUDevice>(),
                 image_data, flow_data, output_data);
  }
};

class BackwardWarpOpGrad : public OpKernel {
public:
  explicit BackwardWarpOpGrad(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& original_images = context->input(1);
    const Tensor& original_flows = context->input(2);

    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, original_flows.shape(),
                                                     &output));

    typename TTypes<float, 4>::ConstTensor input_data = input.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor flow_data = original_flows.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor image_data = original_images.tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    BackwardWarpGrad(context->eigen_device<GPUDevice>(),
                     input_data, image_data, flow_data, output_data);
  }
};

REGISTER_OP("BackwardWarp")
  .Input("images: float")
  .Input("flows: float")
  .Output("warped_images: float")
  .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("BackwardWarpGrad")
  .Input("grads: float")
  .Input("original_images: float")
  .Input("original_flows: float")
  .Output("output: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(2));
    return Status::OK();
  });

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("BackwardWarp").Device(DEVICE_GPU), BackwardWarpOp);
REGISTER_KERNEL_BUILDER(Name("BackwardWarpGrad").Device(DEVICE_GPU), BackwardWarpOpGrad);

#endif // GOOGLE_CUDA
