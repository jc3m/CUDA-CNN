
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_NUM 64
#define FEATURE_NUM 64
#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{
/*
	  ____.    ____.  .__                    ___.   .__  __         .__
    |    |   |    |  |__| ______  _____     \_ |__ |__|/  |_  ____ |  |__
    |    |   |    |  |  |/  ___/  \__  \     | __ \|  \   __\/ ___\|  |  \
/\__|    /\__|    |  |  |\___ \    / __ \_   | \_\ \  ||  | \  \___|   Y  \
\________\________|  |__/____  >  (____  /   |___  /__||__|  \___  >___|  /
									\/        \/        \/              \/     \/
*/
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    int img = blockIdx.x;										// b
    int feature = blockIdx.y;									// m
    int width = threadIdx.x;
    int height = threadIdx.y;

    if (height < H && width < W) {
        float sum = 0.0f;
        // Sum over all feature maps
        for (int c = 0; c < C; ++c) {
            // Single convolution step: KxK filter
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    sum += x4d(img, c, height + p, width + q) * k4d(feature, c, p, q);
                }
            }
        }
        y4d(img, feature, height, width) = sum;
    }

    #undef y4d
    #undef x4d
    #undef k4d
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; // Number of Images, y.shape_[0] should be the same
    const int M = y.shape_[1]; // Number of the output feature maps
    const int C = x.shape_[1]; // Number of input feature maps
    const int H = x.shape_[2]; // Height of an input feature map
    const int W = x.shape_[3]; // Width of an input feature map
    const int K = w.shape_[3]; // Side length of a filter

    // Set the kernel dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    dim3 gridDim = {
        (unsigned int)B,
        (unsigned int)M,
        1
    };
    dim3 blockDim = {
        (unsigned int)W_out,
        (unsigned int)H_out,
        1
    };

    // Call the kernel
    forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}


/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
