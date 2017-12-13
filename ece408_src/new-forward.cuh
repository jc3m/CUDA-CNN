
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)

// Fast ceil macro that doesn't require float casting
#define int_ceil(x,y) (x + y - 1) / y
#define my_min(x,y) ((x > y) ? y : x)

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
__global__ void matrixMultiplyShared(float *arr_A, float *arr_B, float *arr_C, const int B) {
    /*****************/
    /* Shared Memory */
    /*****************/
    __shared__ float filters[FILTER_SIZE][M]; // Transposed filters
    __shared__ float X[H][W];

    /********************/
    /* Kernel Registers */
    /********************/
    float thread_output[THREADY_DIVISOR][THREADX_DIVISOR];
    unsigned int linear_idx = threadIdx.y * blockDim.x + threadIdx.x; //Linearized index
    int img = blockIdx.x;

    /*******************************************/
    /* Loading all of arr_A into shared memory */
    /*******************************************/
    if (linear_idx < M * FILTER_SIZE / 2) {
        // Each thread will load 2 elements
        unsigned int filter_y_base = linear_idx % FILTER_SIZE;
        unsigned int filter_x_base = linear_idx / FILTER_SIZE;
        // Transposing base
        filters[filter_y_base][filter_x_base] = arr_A[linear_idx];
        filters[filter_y_base][filter_x_base + M / 2] = arr_A[linear_idx + M * FILTER_SIZE / 2];
    }

    /*******************************************/
    /* Loading all of arr_B into shared memory */
    /*******************************************/
    if (linear_idx >= THREADS_PER_BLOCK - (INPUT_FEATURE_SIZE / 2)) {
        unsigned int normalized_index = linear_idx - (INPUT_FEATURE_SIZE / 2);
        unsigned int input_x_base = normalized_index % W;
        unsigned int input_y_base = normalized_index / W;
        X[input_y_base][input_x_base] = arr_B[normalized_index];
        X[input_y_base + H / 2][input_x_base] = arr_B[normalized_index + INPUT_FEATURE_SIZE / 2];
    }

    __syncthreads();
    #pragma unroll
    for (unsigned int i = 0; i < THREADX_DIVISOR; i++) {
        #pragma unroll
        for (unsigned int k = 0; k < FILTER_SIZE; k++) {
            // Value from the unrolled matrix
            int h_unroll = i * BLOCK_DIM_X + threadIdx.x;
            int w_out = h_unroll % W_out;
            int h_out = h_unroll / W_out;
            int q = k % K;
            int p = k / K;

            float unrolled_val = X[h_out + p, w_out + q];
            #pragma unroll
            for (unsigned int j = 0; j < THREADY_DIVISOR; j++) {
                float filter_val = filters[k][j * BLOCK_DIM_Y + threadIdx.y];
                thread_output[j][i] += filter_val * unrolled_val;
            }
        }
    }

    /******************************/
    /* Writeback to global memory */
    /******************************/
    unsigned int output_base = img * H_out * W_out * M + threadIdx.y * OUTPUT_FEATURE_SIZE + threadIdx.x;
    #pragma unroll
    for (unsigned int i = 0; i < THREADY_DIVISOR; i++) {
        #pragma unroll
        for (unsigned int j = 0; j < THREADX_DIVISOR; j++) {
            arr_C[output_base + (i * BLOCK_DIM_Y) * OUTPUT_FEATURE_SIZE  + j * BLOCK_DIM_X] = thread_output[i][j];
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

    dim3 blockDim = { BLOCK_DIM_X, BLOCK_DIM_Y, 1 };
    dim3 gridDim = { B, 1, 1 };

    matrixMultiplyShared<<<gridDim, blockDim, 0, s>>>(w.dptr_, x.dptr_, y.dptr_, B);
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
