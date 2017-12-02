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

#include "correlation_op.h"

typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow;

void Correlation(const GPUDevice& d,
                 typename TTypes<float, 4>::ConstTensor input_0,
                 typename TTypes<float, 4>::ConstTensor input_1,
                 typename TTypes<float, 4>::Tensor output,
                 typename TTypes<float, 4>::Tensor padded_0,
                 typename TTypes<float, 4>::Tensor padded_1,
                 CorrelationState params);

void CorrelationGrad(const GPUDevice& d,
                     typename TTypes<float, 4>::ConstTensor input_grad,
                     typename TTypes<float, 4>::ConstTensor padded_0,
                     typename TTypes<float, 4>::ConstTensor padded_1,
                     typename TTypes<float, 4>::Tensor output_grad_0,
                     typename TTypes<float, 4>::Tensor output_grad_1,
                     CorrelationState params);

class CorrelationOp : public OpKernel {
public:
  explicit CorrelationOp(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_0 = context->input(0);
    const Tensor& input_1 = context->input(1);

    OP_REQUIRES(context, input_0.shape() == input_1.shape(),
                errors::InvalidArgument("Input shapes have to be the same"));

    typename TTypes<float, 4>::ConstTensor input_0_data = input_0.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor input_1_data = input_1.tensor<float, 4>();

    const int batch = input_0_data.dimension(0);
    const int in_channels = input_0_data.dimension(1);
    const int in_height = input_0_data.dimension(2);
    const int in_width = input_0_data.dimension(3);

    CorrelationState st(attrs, in_height, in_width, in_channels);

    OP_REQUIRES(context, st.out_width * st.out_height > 0,
                errors::InvalidArgument("Invalid correlation settings"));

    Tensor* output = NULL;
    TensorShape output_shape({batch, st.out_channels, st.out_height, st.out_width});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    Tensor* padded_0 = NULL;
    Tensor* padded_1 = NULL;
    TensorShape padded_shape({batch, st.padded_height, st.padded_width, in_channels});
    OP_REQUIRES_OK(context, context->allocate_output(1, padded_shape, &padded_0));
    OP_REQUIRES_OK(context, context->allocate_output(2, padded_shape, &padded_1));

    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor padded_0_data = padded_0->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor padded_1_data = padded_1->tensor<float, 4>();

    Correlation(context->eigen_device<GPUDevice>(),
                input_0_data, input_1_data, output_data,
                padded_0_data, padded_1_data,
                st);
  }

private:
  CorrelationAttrs attrs;
};

class CorrelationOpGrad : public OpKernel {
public:
  explicit CorrelationOpGrad(OpKernelConstruction* context)
  : OpKernel(context), attrs(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_grad = context->input(0);
    const Tensor& input_0 = context->input(1);
    const Tensor& input_1 = context->input(2);
    const Tensor& padded_0 = context->input(3);
    const Tensor& padded_1 = context->input(4);

    typename TTypes<float, 4>::ConstTensor input_grad_data = input_grad.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor input_0_data = input_0.tensor<float, 4>();
    //typename TTypes<float, 4>::ConstTensor input_1_data = input_1.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor padded_0_data = padded_0.tensor<float, 4>();
    typename TTypes<float, 4>::ConstTensor padded_1_data = padded_1.tensor<float, 4>();

    const int in_channels = input_0_data.dimension(1);
    const int in_height = input_0_data.dimension(2);
    const int in_width = input_0_data.dimension(3);

    CorrelationState st(attrs, in_height, in_width, in_channels);

    Tensor* output_grad_0 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_0.shape(),
                                                     &output_grad_0));
    Tensor* output_grad_1 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, input_0.shape(),
                                                     &output_grad_1));

    typename TTypes<float, 4>::Tensor output_grad_0_data = output_grad_0->tensor<float, 4>();
    typename TTypes<float, 4>::Tensor output_grad_1_data = output_grad_1->tensor<float, 4>();

    CorrelationGrad(context->eigen_device<GPUDevice>(),
                    input_grad_data,
                    padded_0_data, padded_1_data,
                    output_grad_0_data, output_grad_1_data,
                    st);
  }
private:
  CorrelationAttrs attrs;
};

using shape_inference::DimensionHandle;;

REGISTER_OP("Correlation")
  .Input("input_0: float")
  .Input("input_1: float")
  .Attr("kernel_size: int = 1")
  .Attr("max_displacement: int = 20")
  .Attr("pad: int = 20")
  .Attr("stride_1: int = 1")
  .Attr("stride_2: int = 2")
  .Output("correlation: float")
  .Output("padded_0: float")
  .Output("padded_1: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    CorrelationAttrs attrs;
    c->GetAttr("kernel_size", &attrs.kernel_size);
    c->GetAttr("max_displacement", &attrs.max_displacement);
    c->GetAttr("pad", &attrs.pad_size);
    c->GetAttr("stride_1", &attrs.stride_1);
    c->GetAttr("stride_2", &attrs.stride_2);

    DimensionHandle batch = c->Dim(c->input(0), 0);

    //padded_height = in_height + 2 * pad_size;
    //padded_width = in_width + 2 * pad_size;
    //kernel_radius = (kernel_size - 1) / 2;
    //border_size = max_displacement + kernel_radius;
    int neighborhood_grid_radius = attrs.max_displacement / attrs.stride_2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
    //out_width = ceil((float)(padded_width - border_size *2) / (float)stride_1);
    //out_height = ceil((float)(padded_height - border_size *2) / (float)stride_1);
    int out_channels = neighborhood_grid_width * neighborhood_grid_width;

    // TODO: support passing on output width and height

    c->set_output(0, c->MakeShape({batch, out_channels, c->UnknownDim(), c->UnknownDim()}));
    return Status::OK();
  });

REGISTER_OP("CorrelationGrad")
  .Input("input_grad: float")
  .Input("original_input_0: float")
  .Input("original_input_1: float")
  .Input("padded_0: float")
  .Input("padded_1: float")
  .Attr("kernel_size: int = 1")
  .Attr("max_displacement: int = 20")
  .Attr("pad: int = 20")
  .Attr("stride_1: int = 1")
  .Attr("stride_2: int = 2")
  .Output("output_grad_0: float")
  .Output("output_grad_1: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(1));
    c->set_output(1, c->input(2));
    return Status::OK();
  });

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("Correlation").Device(DEVICE_GPU), CorrelationOp);
REGISTER_KERNEL_BUILDER(Name("CorrelationGrad").Device(DEVICE_GPU), CorrelationOpGrad);

#endif // GOOGLE_CUDA
