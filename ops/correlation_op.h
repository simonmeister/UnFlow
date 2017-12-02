#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

struct CorrelationAttrs {
  CorrelationAttrs(OpKernelConstruction* c) {
    OP_REQUIRES_OK(c, c->GetAttr("kernel_size", &kernel_size));
    OP_REQUIRES_OK(c, c->GetAttr("max_displacement", &max_displacement));
    OP_REQUIRES_OK(c, c->GetAttr("pad", &pad_size));
    OP_REQUIRES_OK(c, c->GetAttr("stride_1", &stride_1));
    OP_REQUIRES_OK(c, c->GetAttr("stride_2", &stride_2));

    OP_REQUIRES(c, kernel_size % 2 != 0,
                errors::InvalidArgument("kernel_size must be odd"));
  }
  CorrelationAttrs() {}

  int pad_size;
  int stride_1;
  int stride_2;
  int max_displacement;
  int kernel_size;
};

struct CorrelationState {
  CorrelationState(CorrelationAttrs attrs, int in_height, int in_width, int in_channels) {
    pad_size = attrs.pad_size;
    stride_1 = attrs.stride_1;
    stride_2 = attrs.stride_2;
    max_displacement = attrs.max_displacement;
    kernel_size = attrs.kernel_size;

    padded_height = in_height + 2 * pad_size;
    padded_width = in_width + 2 * pad_size;

    // Compute size of unreachable border region (on each side)
    kernel_radius = (kernel_size - 1) / 2;
    border_size = max_displacement + kernel_radius;

    // Given a center position in image 1, how many displaced positions in -x / +x
    // direction do we consider in image 2 (neighborhoodGridWidth):
    neighborhood_grid_radius = max_displacement / stride_2;
    neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

    out_width = ceil((float)(padded_width - border_size *2) / (float)stride_1);
    out_height = ceil((float)(padded_height - border_size *2) / (float)stride_1);
    // Top Channels amount to displacement combinations in X and Y direction:
    out_channels = neighborhood_grid_width * neighborhood_grid_width;
  }

  int pad_size;
  int stride_1;
  int stride_2;
  int kernel_radius;
  int max_displacement;
  int kernel_size;
  int neighborhood_grid_radius;
  int neighborhood_grid_width;
  int padded_height;
  int padded_width;
  int border_size;
  int out_height;
  int out_width;
  int out_channels;
};
