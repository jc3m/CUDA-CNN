
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_NUM 64
#define FEATURE_NUM 64
#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)
#define CUDA_MAX_NUM_THREADS 1024

#define TILE_WIDTH 16
#define MIN_BATCH_SIZE 4

//Fast ceil macro that doesn't require float casting
#define int_ceil(x,y) (x + y - 1) / y
#define my_min(x,y) ((x > y) ? y : x)

//Network constants
//B: 10000, M: 50, C: 1, H: 28, W: 28, K: 5
#define M 50    // Number of the output feature maps
#define C 1     // Number of input feature maps
#define H 28    // Height of an input feature map
#define W 28    // Width of an input feature map
#define K 5     // Side length of a filter

#define H_out (H - K + 1)
#define W_out (W - K + 1)

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
__global__ void matrixMultiplyShared(float *A, float *x, float *Carr, const int b, const int B) {
    #define numARows M
    #define numAColumns C * K * K
    #define numBRows C * K * K
    #define numBColumns H_out * W_out
    #define numCRows M
    #define numCColumns H_out * W_out
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	 int img = b*MIN_BATCH_SIZE + blockIdx.z;
    float Pvalue = 0;

    for (int m = 0; m < ceil(numAColumns/(float)TILE_WIDTH); m++) {
        // Parallel memory reads
        if ((Row < numCRows) && (m * TILE_WIDTH + threadIdx.x < numAColumns)) {
            subTileA[threadIdx.y][threadIdx.x] = A[Row*numAColumns + m*TILE_WIDTH + threadIdx.x];
        } else {
            subTileA[threadIdx.y][threadIdx.x] = 0;
        }

        if ((m * TILE_WIDTH + threadIdx.y < numBRows) && (Col < numCColumns)) {
            // subTileB[threadIdx.y][threadIdx.x] = B[m*TILE_WIDTH + threadIdx.y][]
            int w_unroll = m * TILE_WIDTH + threadIdx.y;
            int h_unroll = Col;
            int w_out = h_unroll % W_out;
            int h_out = h_unroll / W_out;
            int q = w_unroll % K;
            int p = (w_unroll / K) % K;
            int c = w_unroll / (K * K);
            // subTileB[threadIdx.y][threadIdx.x] = B[(m*TILE_WIDTH + threadIdx.y)*numBColumns + Col];
            subTileB[threadIdx.y][threadIdx.x] = x4d(img, c, h_out + p, w_out + q);
        } else {
            subTileB[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Tile calculation
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];
        }

        __syncthreads();
    }
    if ((Row < numCRows) && (Col < numCColumns)) {
        Carr[img * numCColumns * numCRows + Row * numCColumns + Col] = Pvalue;
    }
    #undef numARows
    #undef numAColumns
    #undef numBRows
    #undef numBColumns
    #undef numCRows
    #undef numCColumns
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

    for (int b = 0; b < int_ceil(B, MIN_BATCH_SIZE); b++) {
        //float *xb = &(x.dptr_[b * C * H * W]);
        //float *yb = &(y.dptr_[b * M * H_out * W_out]);

        // int k_rows = M;
        // int k_cols = C * K * K;
        // int x_rows = C * K * K;
        // int x_cols = H_out * W_out;
        int y_rows = M;
        int y_cols = H_out * W_out;

        dim3 blockDim = { TILE_WIDTH, TILE_WIDTH, 1 };
        dim3 gridDim = {
            (unsigned int)ceil((float)y_cols / (float)TILE_WIDTH),
            (unsigned int)ceil((float)y_rows / (float)TILE_WIDTH),
            (unsigned int)my_min((B - b*MIN_BATCH_SIZE), MIN_BATCH_SIZE)
        };
        matrixMultiplyShared<<<gridDim, blockDim, 0, s>>>(w.dptr_, x.dptr_, y.dptr_, b, B);
        cudaDeviceSynchronize();
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
