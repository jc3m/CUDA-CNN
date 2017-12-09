
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define IMG_NUM 64
#define FEATURE_NUM 64
#define IMG_SIDE_LENGTH 24
#define IMG_AREA (IMG_SIDE_LENGTH * IMG_SIDE_LENGTH)
#define CUDA_MAX_NUM_THREADS 1024

#define TILE_WIDTH 16
#define MIN_BATCH_SIZE 8

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

__device__ int3 get_b_idxs(int m, int ty, int Col)
{
  int w_unroll = m * TILE_WIDTH + ty;
  int h_unroll = Col;
  int w_out = h_unroll % W_out;
  int h_out = h_unroll / W_out;
  int q = w_unroll % K;
  int p = (w_unroll / K) % K;
  int c = w_unroll / (K * K);
  return make_int3(c, h_out + p, w_out + q);
}

__global__ void matrixMultiplyShared(float *A, float *x, float *Carr, const int b, const int B) {
    #define numARows M
    #define numAColumns C * K * K
    #define numBRows C * K * K
    #define numBColumns H_out * W_out
    #define numCRows M
    #define numCColumns H_out * W_out
    #define MUL_LOOP_ITER int_ceil(numAColumns, TILE_WIDTH)
    #define size_A (numARows * numAColumns)
    #define size_B (MIN_BATCH_SIZE * C * H * W)

    //Since each kernel is a single image, we can coalesce by having all threads work together to load the image's data into shared memory to start
    __shared__ float shared_A[size_A];
    __shared__ float shared_B[size_B];

	 int img = b*MIN_BATCH_SIZE + blockIdx.z;
    int total_threads = TILE_WIDTH * TILE_WIDTH * blockDim.x * blockDim.y * blockDim.z;
    int width = TILE_WIDTH * blockDim.x;
    int lin_idx_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int lin_idx_y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int lin_idx = (width * width) * blockIdx.z + lin_idx_y * width + lin_idx_x;

    int i = lin_idx;
    while(i < size_B) {
      shared_B[i] = x[i];
      i += total_threads;
    }

    i = lin_idx;
    while(i < size_A) {
      shared_A[i] = A[i];
      i += total_threads;
    }

    __syncthreads();

    for(int i = 0; i < int_ceil(numCColumns,TILE_WIDTH); i++) {
      for(int j = 0; j < int_ceil(numCRows,TILE_WIDTH); j++) {
         int Row = j * width + blockIdx.y * TILE_WIDTH + threadIdx.y;
         int Col = i * width + blockIdx.x * TILE_WIDTH + threadIdx.x;

         float p_value = 0.0f;
         for(int m = 0; m < numAColumns; m++) {
            int3 idxs_B = get_b_idxs(m, threadIdx.y, Col);
            float B_val = shared_B[C*H*W*blockIdx.z + H*W*idxs_B.x + H*idxs_B.y + idxs_B.z];
            //float B_val = shared_B[idxs_B.x][idxs_B.y][idxs_B.z];
            float A_val = shared_A[Row*numAColumns + Col];
            p_value += A_val * B_val;
         }
         if(Row < numCRows && Col < numCColumns)
            Carr[img * numCColumns * numCRows + Row * numCColumns + Col] = p_value;
      }
   }

   /*

       //Read global memory coalesced
       for(int i = 0; i < int_ceil(H,TILE_WIDTH); i++) {
         for(int j = 0; j < int_ceil(W,TILE_WIDTH); j++) {
            int Row = (blockIdx.y + blockDim.y * j) * TILE_WIDTH + threadIdx.y;
            int Col = (blockIdx.x + blockDim.y * i) * TILE_WIDTH + threadIdx.x;

            if(Col < W && Row < H) {
              shared_B[blockIdx.z][0][Row][Col] = x4d(img, 0, Row, Col);
              shared_B[blockIdx.z][1][Row][Col] = x4d(img, 1, Row, Col);
            }
         }
      }

       for(int i = 0; i < int_ceil(numAColumns,TILE_WIDTH); i++) {
         for(int j = 0; j < int_ceil(numARows,TILE_WIDTH); j++) {
            int Row = (blockIdx.y + blockDim.y * j) * TILE_WIDTH + threadIdx.y;
            int Col = (blockIdx.x + blockDim.y * i) * TILE_WIDTH + threadIdx.x;

            if(Col < numAColumns && Row < numARows)
              shared_A[Row][Col] = A[Row * numAColumns + Col];
         }
       }
       __syncthreads();*/
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
        int y_rows = M;
        int y_cols = H_out * W_out;

        dim3 blockDim = { TILE_WIDTH, TILE_WIDTH, 1 };
        dim3 gridDim = {
            (unsigned int)int_ceil(y_cols, TILE_WIDTH),
            (unsigned int)int_ceil(y_rows, TILE_WIDTH),
            (unsigned int)my_min((B - b*MIN_BATCH_SIZE), MIN_BATCH_SIZE)
        };
        matrixMultiplyShared<<<gridDim, blockDim, 0, s>>>(w.dptr_, &((x.dptr_)[(C * K * K) * b*MIN_BATCH_SIZE]), y.dptr_, b, B);

    }
	 cudaDeviceSynchronize();

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
