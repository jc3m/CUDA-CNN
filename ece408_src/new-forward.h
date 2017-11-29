#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet {
namespace op {

// This function is called by new-inl.h
// Any code you write should be executed by this function
template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x,
             const mshadow::Tensor<cpu, 4, DType> &k) {
    // Modify this function to implement the forward pass described in Chapter 16.
    // The code in 16 is for a single image.
    // We have added an additional dimension to the tensors to support an entire mini-batch
    // The goal here is to be correct, not fast (this is the CPU implementation.)

    const int B = x.shape_[0]; // Number of Images, y.shape_[0] should be the same
    const int M = y.shape_[1]; // Number of the output feature maps
    const int C = x.shape_[1]; // Number of input feature maps
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = k.shape_[3];

    // For each image in the batch
    for (int b = 0; b < B; ++b) {
        // For each output feature map
        for (int m = 0; m < M; ++m) {
            // For each output element
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    y[b][m][h][w] = 0;
                    // Sum over all feature maps
                    for (int c = 0; c < C; ++c) {
                        // Single convolution step: KxK filter
                        for (int p = 0; p < K; ++p) {
                            for (int q = 0; q < K; ++q) {
                                y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                            }
                        }
                    }
                }
            }
        }
    }
}
}
}

#endif
