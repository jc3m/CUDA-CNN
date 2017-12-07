#ifndef MATRIX_MUL_CUH
#define MATRIX_MUL_CUH

#define TILE_WIDTH 32

//     ____.    ____. .__                                               .__
//    |    |   |    | |__| ______ _____      _____ _____    ______ _____|__|__  __ ____   ______  __ __  ______ _________.__.
//    |    |   |    | |  |/  ___/ \__  \    /     \\__  \  /  ___//  ___/  \  \/ // __ \  \____ \|  |  \/  ___//  ___<   |  |
///\__|    /\__|    | |  |\___ \   / __ \_ |  Y Y  \/ __ \_\___ \ \___ \|  |\   /\  ___/  |  |_> >  |  /\___ \ \___ \ \___  |
//\________\________| |__/____  > (____  / |__|_|  (____  /____  >____  >__| \_/  \___  > |   __/|____//____  >____  >/ ____|  

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
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
    C[Row * numCColumns + Col] = Pvalue;
}

#endif
