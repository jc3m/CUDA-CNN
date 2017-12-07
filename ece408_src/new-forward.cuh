
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_NUM 64
#define FEATURE_NUM 64
#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)
#define CUDA_MAX_NUM_THREADS 1024

#define TILE_WIDTH 16

//Fast ceil macro that doesn't require float casting
#define int_ceil(x,y) (x + y - 1) / y;

//Network constants
//B: 10000, M: 50, C: 1, H: 28, W: 28, K: 5
#define M 50
#define C 1
#define H 28
#define W 28
#define K 5

#define H_out (H - K + 1)
#define W_out (W - K + 1)
// #define W_unroll (C * K * K)
// #define H_unroll (H_out * W_out)

#define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
#define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
#define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

#include <mxnet/base.h>
// #include "matrix_mul.cuh"

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
__global__ void matrixMultiplyShared(float *A, float *B, float *Carr,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float Pvalue = 0;

  for (int m = 0; m < ceil(numAColumns/(float)TILE_WIDTH); m++) {
    // Parallel memory reads
    if((Row < numCRows) && (m * TILE_WIDTH + threadIdx.x < numAColumns)) {
      subTileA[threadIdx.y][threadIdx.x] = A[Row*numAColumns + m*TILE_WIDTH + threadIdx.x];
    } else {
      subTileA[threadIdx.y][threadIdx.x] = 0;
    }

    if((m * TILE_WIDTH + threadIdx.y < numBRows) && (Col < numCColumns)) {
      subTileB[threadIdx.y][threadIdx.x] = B[(m*TILE_WIDTH + threadIdx.y)*numBColumns + Col];
    } else {
      subTileB[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    // Tile calculation
    for (int k = 0; k < TILE_WIDTH; k++)
      Pvalue += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];

    __syncthreads();
  }
  if((Row < numCRows) && (Col < numCColumns))
    Carr[Row * numCColumns + Col] = Pvalue;
}

__global__ void unroll_kernel(float *x, float *X_unroll) {
    // The following kernel copies a KxK section of X that will be used to compute
    // exactly one output element in the convolution
    int t = blockIdx.x * CUDA_MAX_NUM_THREADS + threadIdx.x;

    int W_unroll = H_out * W_out;           // Number of elements in an output feature map
    // int threadsPerImage = C * W_unroll;     // C * H_out * W_out
    // int totalThreads = B * threadsPerImage; // B * C * H_out * W_out

    if (t < W_unroll) {
        // It'd be cool to speed these up with some inline
        // int b = t / threadsPerImage;            // The image # in the batch
        // int i = t % threadsPerImage;            // InputFeatureMap/OutputElement combo in the given image
        int c = t / W_unroll;                   // The given feature map
        int s = t % W_unroll;                   // Linearized index in the output feature map
        int h_out = s / W_out;                  // Height in the output feature map
        int w_out = s % W_out;                  // Width in the output feature map

        int h_unroll = h_out * W_out + w_out;
        int w_base = c * K * K;

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int w_unroll = w_base + p * K + q;

                // X_unroll[b, w_unroll, h_unroll] = X[b, c, h_out + p, w_out + q];
                #define X_unroll3d(i2,i1,i0) X_unroll[(i2)*(H_out * W_out * C * K * K) + (i1)*(H_out * W_out) + i0]
                #define X_unroll2d(i1,i0) X_unroll[i1*(H_out * W_out) + i0]
                // X_unroll3d(b, w_unroll, h_unroll) = x4d(b, c, h_out + p, w_out + q);
                X_unroll2d(w_unroll, h_unroll) = x4d(0, c, h_out + p, w_out + q);
            }
        }
    }
}

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
}

static void unroll(float *X, float *X_unroll, cudaStream_t s) {
    unsigned int num_threads = C * H_out * W_out;
    unsigned int num_blocks = int_ceil(num_threads, CUDA_MAX_NUM_THREADS);

    unroll_kernel<<<num_blocks, CUDA_MAX_NUM_THREADS, 0, s>>>(X, X_unroll);
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
    // const int M = y.shape_[1]; // Number of the output feature maps
    // const int C = x.shape_[1]; // Number of input feature maps
    // const int H = x.shape_[2]; // Height of an input feature map
    // const int W = x.shape_[3]; // Width of an input feature map
    // const int K = w.shape_[3]; // Side length of a filter

    // H_out and W_out should both be 24

    // Unroll
    float *X_unroll;
    cudaMalloc(&X_unroll, sizeof(float) * C * K * K * H_out * W_out);

    // Call the kernel
    // forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    for (int b = 0; b < B; b++) {
        float *xb = &(x.dptr_[b * C * H * W]);
        unroll(xb, X_unroll, s);
        cudaDeviceSynchronize();

        float *yb = &(y.dptr_[b * M * H_out * W_out]);

        int k_rows = M;
        int k_cols = C * K * K;
        int x_rows = C * K * K;
        int x_cols = H_out * W_out;
        int y_rows = M;
        int y_cols = H_out * W_out;

        dim3 blockDim = { TILE_WIDTH, TILE_WIDTH, 1 };
        dim3 gridDim = {
            (unsigned int)ceil((float)y_cols / (float)TILE_WIDTH),
            (unsigned int)ceil((float)y_rows / (float)TILE_WIDTH),
            1
        };
        matrixMultiplyShared<<<gridDim, blockDim>>>(w.dptr_, X_unroll, yb, k_rows, k_cols, x_rows, x_cols, y_rows, y_cols);
        cudaDeviceSynchronize();
    }

    cudaFree(X_unroll);

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
