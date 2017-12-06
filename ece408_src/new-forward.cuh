
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_NUM 64
#define FEATURE_NUM 64
#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)
#define CUDA_MAX_NUM_THREADS 1024

//Fast ceil macro that doesn't require float casting
#define int_ceil(x,y) (x + y - 1) / y;

//Network constants
//B: 10000, M: 50, C: 1, H: 28, W: 28, K: 5
#define M 50
#define C 1
#define H 28
#define W 28
#define K 5

__constant__ float k_const[][][][];

#define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
#define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
#define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

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
__device__ void unroll_kernel(const int C, const int H, const int W, const int K, float *X, float *X_unroll, const int H_out, const int W_out) {
    int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;
    int W_unroll = H_out * W_out;

    if(t < C * W_unroll) {
        //It'd be cool to speed these up with some inline
        int c = t / W_unroll;
        int s = t % W_unroll;
        int h_out = s / W_out;
        int w_out = s % W_out;

        int H_unroll = h_out * W_out + w_out;
        int W_base = c * K * K;

        for(int p = 0; p < K; p++) {
            int w_unroll = w_base + p * K + q;
            X_unroll[h_unroll, w_unroll] = X[c, h_out + p, w_out + q];
        }
    }
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;


}

static void unroll(const int C, const int H, const int W, const int K, float *X, float *X_unroll, const int H_out, const int W_out) {
    unsigned int num_threads = C * H_out * W_out;
    unsigned int num_blocks = int_ceil(num_threads, CUDA_MAX_NUM_THREADS);

    unroll_kernel<<<num_blocks, CUDA_MAX_NUM_THREADS>>>(C, H, W, K, X, X_unroll);
}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {

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
    unsigned int W_grid = int_ceil(W_out,IMG_SIDE_LENGTH);
    unsigned int H_grid = int_ceil(H_out,IMG_SIDE_LENGTH);
    unsigned int Z = W_grid * H_grid;
    dim3 gridDim = {
        (unsigned int)B,
        (unsigned int)M,
        (unsigned int)Z
    };
    dim3 blockDim = { IMG_SIDE_LENGTH, IMG_SIDE_LENGTH, 1 };

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
