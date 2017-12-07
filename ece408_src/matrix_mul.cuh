#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

#define TILE_WIDTH 16

#define numARows
#define numAColumns
#define numBRows
#define numBColumns
#define numCRows
#define numCColumns

//     ____.    ____. .__                                               .__
//    |    |   |    | |__| ______ _____      _____ _____    ______ _____|__|__  __ ____   ______  __ __  ______ _________.__.
//    |    |   |    | |  |/  ___/ \__  \    /     \\__  \  /  ___//  ___/  \  \/ // __ \  \____ \|  |  \/  ___//  ___<   |  |
///\__|    /\__|    | |  |\___ \   / __ \_ |  Y Y  \/ __ \_\___ \ \___ \|  |\   /\  ___/  |  |_> >  |  /\___ \ \___ \ \___  |
//\________\________| |__/____  > (____  / |__|_|  (____  /____  >____  >__| \_/  \___  > |   __/|____//____  >____  >/ ____|

/*
	How to optimize matrix multiplication furthur:
	-Have each thread compute multiple outputs
	-Check on bank conflicts
	-Double buffering
	-Using wider loads from shared memory
	https://stackoverflow.com/questions/30703190/faster-matrix-multiplication-in-cuda
*/


// Compute C = A * B
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  //@@ Insert code to implement matrix multiplication here
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  float Pvalue = 0;

  //#pragma unroll
  for(int m = 0; m < ceil(numAColumns/(float)TILE_WIDTH); m++)
  {
    // Parallel memory reads
    subTileA[threadIdx.y][threadIdx.x] = ((Row < numCRows) && (m * TILE_WIDTH + threadIdx.x < numAColumns)) ?
														A[Row*numAColumns + m*TILE_WIDTH + threadIdx.x]   : 0;

    subTileB[threadIdx.y][threadIdx.x] = ((m * TILE_WIDTH + threadIdx.y < numBRows) && (Col < numCColumns)) ?
														B[(m*TILE_WIDTH + threadIdx.y)*numBColumns + Col] : 0;

    __syncthreads();

    // Tile calculation
    //for(int k = 0; k < TILE_WIDTH; k++)
      //Pvalue += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];
	 //#pragma unroll
	 for(int k = 0; k < TILE_WIDTH; k+=4)
	 {
		float4 sub_sum = make_float4(subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x],
								subTileA[threadIdx.y][k+1] * subTileB[k+1][threadIdx.x],
								subTileA[threadIdx.y][k+2] * subTileB[k+2][threadIdx.x],
								subTileA[threadIdx.y][k+3] * subTileB[k+3][threadIdx.x]);
	   Pvalue += sub_sum.x + sub_sum.y + sub_sum.z + sub_sum.w;
	 }

    __syncthreads();
  }
  if((Row < numCRows) && (Col < numCColumns))
    C[Row * numCColumns + Col] = Pvalue;
}

#endif
