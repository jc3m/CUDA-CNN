
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)

// Fast ceil macro that doesn't require float casting
#define int_ceil(x,y) (x + y - 1) / y
#define my_min(x,y) ((x > y) ? y : x)

#define __dankthreads __syncthreads

// Network constants
// B: 10000, M: 50, C: 1, H: 28, W: 28, K: 5
#define M 50    // Number of the output feature maps
#define C 1     // Number of input feature maps
#define H 28    // Height of an input feature map
#define W 28    // Width of an input feature map
#define K 5     // Side length of a filter

#define H_out (H - K + 1)
#define W_out (W - K + 1)

#define INPUT_FEATURE_SIZE (H * W)
#define OUTPUT_FEATURE_SIZE (H_out * W_out)
#define FILTER_SIZE (K * K)
#define TOTAL_FILTER_SIZE (M * FILTER_SIZE)

// We'll have each thread compute THEADX_DIVISOR * THREADY_DIVISOR
// elements in the final output matrix
#define THREADX_DIVISOR 6
#define THREADY_DIVISOR 5

#define BLOCK_DIM_X (OUTPUT_FEATURE_SIZE / THREADX_DIVISOR)
#define BLOCK_DIM_Y (M / THREADY_DIVISOR)
#define THREADS_PER_BLOCK (BLOCK_DIM_X * BLOCK_DIM_Y)

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
__global__ void matrixMultiplyShared(float *arr_A, float *arr_B, float *arr_C) {
    /*****************/
    /* Shared Memory */
    /*****************/
    __shared__ float filters_shared[TOTAL_FILTER_SIZE]; //filters_shared[M][FILTER_SIZE]
    __shared__ float x_shared[H * W];

    unsigned int linear_idx = threadIdx.y * BLOCK_DIM_X + threadIdx.x;

    /**************************************/
    /* Loading filter into shared memeory */
    /**************************************/
    if (linear_idx < TOTAL_FILTER_SIZE/2) {
        filters_shared[linear_idx] = arr_A[linear_idx];
        filters_shared[linear_idx + TOTAL_FILTER_SIZE/2] = arr_A[linear_idx + TOTAL_FILTER_SIZE/2];
    }

    /**************************************************/
    /* Loading input feature maps into shared memeory */
    /**************************************************/
    #define FIRST_THREAD_X (THREADS_PER_BLOCK - INPUT_FEATURE_SIZE/2)
    if (linear_idx >= FIRST_THREAD_X) {
        unsigned int normalized_idx = linear_idx - FIRST_THREAD_X;
        unsigned int offset = blockIdx.x * INPUT_FEATURE_SIZE;
        x_shared[normalized_idx] = arr_B[offset + normalized_idx];
        x_shared[normalized_idx + INPUT_FEATURE_SIZE/2] = arr_B[offset + normalized_idx + INPUT_FEATURE_SIZE/2];
    }
    #undef FIRST_THREAD_X

    __dankthreads();

    float result;
    volatile unsigned int h_out, w_out, h_unroll, y_out;

    #pragma unroll
    for (int i = 0; i < THREADX_DIVISOR; i++) {
        h_unroll = i * BLOCK_DIM_X + threadIdx.x;
        h_out = h_unroll / W_out;
        w_out = h_unroll % W_out;
        #pragma unroll
        for (int j = 0; j < THREADY_DIVISOR; j++) {
            result = 0.0f;
            y_out = (j * BLOCK_DIM_Y + threadIdx.y);
            #pragma unroll
            for (int p = 0; p < K; p++) {
                #pragma unroll
                for (int q = 0; q < K; q++) {
                    result += x_shared[(h_out + p) * W + w_out + q] * filters_shared[y_out * FILTER_SIZE + p * K + q];
                }
            }
            #define c3d(i1,i2,i3) arr_C[i1 * H_out * W_out * M + i2 * H_out * W_out + i3]
            c3d(blockIdx.x, y_out, h_unroll) = result;
        }
    }

}

/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x,
                         const mshadow::Tensor<gpu, 4, float> &w) {

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const unsigned int B = x.shape_[0]; // Number of Images, y.shape_[0] should be the same

    dim3 gridDim = { B, 1, 1 };
    dim3 blockDim = { BLOCK_DIM_X, BLOCK_DIM_Y, 1 };

    matrixMultiplyShared<<<gridDim, blockDim, 0, s>>>(w.dptr_, x.dptr_, y.dptr_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

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
