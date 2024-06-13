#include <iostream>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(pointer)[0])
/*
 * C = A @ B
 * A,B,C are row-major matrix
 * A:(M,K), B:(K,N), C(M,N)
 *
 */
__global__ void sgemm_v1(float *A, float *B, float *C, const int M, const int N,
                         const int K) {
  // thread info
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  // building block info
  //  1) A thread block calculate a 128*128 sub-matrix in the C matrix
  //  2) A thread calculate a 8*8 sub-matrix in the C matrix
  const int BM = 128;
  const int BN = 128;
  const int BK = 8;
  const int TM = 8;
  const int TN = 8;

  float c_r[TM][TN] = {0.0};

  // The element used to calculate the 128*128 sub-matrix should be stored in
  // the SHM
  __shared__ float s_a[BM][BK];
  __shared__ float s_b[BK][BN];

  // Load elements from global memory to the shared memory
  /*
   *1) Elements number loaded for every thread:
   *   There BM/TM * BN/TN threads and BM*BK elements in A sub-matrix and BK*BN
   *elemens in the B sub-matrix So, every thread load BK*TM*TN/BN=4 elements
   *from A and BK*TM*TN/BM=4 elements from the B 2) There are 8 elemnts in every
   *row A sub-matrix to load and 128 elments in every row of sub-matrix B So,
   *Float4 can be used to speed-up the load
   *
   *
   */
  // 1) local row/col id for every thread
  int shm_a_load_row =
      tid >> 1; // tid/2, row id in the s_a, 2 Float4 in every row
  int shm_a_load_col = (tid & 1) << 2; // tid%2, 0 OR 4, col id in the s_a
  int shm_b_load_row =
      tid >> 5; // tid/32, row id in the s_b, 128/4=32 Float4 in every row
  int shm_b_load_col = (tid & 31) << 2;

  // 2) global row/col id for every thread, need to add the block offset

  int gemm_a_load_row = by * BM + shm_a_load_row;
  int gemm_b_load_col = bx * BN + shm_b_load_col;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int gemm_a_load_col = bk * BK + shm_a_load_col;
    auto gemm_a_load_offset = OFFSET(gemm_a_load_row, gemm_a_load_col, K);
    FLOAT4(&(s_a[shm_a_load_row][shm_a_load_col])) =
        FLOAT4(&(A[gemm_a_load_offset]));
    int gemm_b_load_row = bk * BK + shm_b_load_row;
    auto gemm_b_load_offset = OFFSET(gemm_b_load_row, gemm_b_load_col, N);
    FLOAT4(&(s_b[shm_b_load_row][shm_b_load_col])) =
        FLOAT4(&(B[gemm_b_load_offset]));
    __syncthreads();
    // calculate the C(TM, TN)
    for (int k = 0; k < BK; k++) {
      for (int i = 0; i < TM; i++) {
        auto s_a_row = ty * TM + i;
        for (int j = 0; j < TN; j++) {
          auto s_b_col = tx * TN + j;
          c_r[i][j] += s_a[s_a_row][k] * s_b[k][s_b_col];
        }
      }
    }
    __syncthreads();
  }

  // store the c_r result into the global memory
  int gemm_c_store_row = by * BM + ty * TM;
  int gemm_c_store_col = bx * BM + tx * TN;
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      C[OFFSET(gemm_c_store_row + i, gemm_c_store_col + j, N)] = c_r[i][j];
    }
  }
}


void sgemm_cpu(float *A, float *B, float *C, const int M, const int N,
               const int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0;
      for (int k = 0; k < K; k++) {
        sum += A[OFFSET(i, k, K)] * B[OFFSET(k, j, N)];
      }
      C[OFFSET(i, j, K)] = sum;
    }
  }
}
bool test_acc(float *A, float *B, float *C, const int M, const int N,
              const int K, float abl = 2e-4) {

  float *h_C, *d_A, *d_B, *d_C;
  auto size_a = M * K * sizeof(float);
  auto size_b = K * N * sizeof(float);
  auto size_c = M * N * sizeof(float);
  cudaMalloc(&d_A, size_a);
  cudaMalloc(&d_B, size_b);
  cudaMalloc(&d_C, size_c);
  h_C = (float *)malloc(size_c);

  cudaMemcpy(d_A, A, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size_b, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, size_c, cudaMemcpyHostToDevice);

  // Get the GPU result
  auto BM = 128;
  auto BN = 128;
  auto TM = 8;
  auto TN = 8;
  dim3 block(BM / TM, BN / TN);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  sgemm_v1<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

  cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost);

  // get cpu result
  sgemm_cpu(A, B, C, M, N, K);

  // compare the result between cuda and cpu
  for (int i = 0; i < M; i < i++) {
    for (int j = 0; j < N; j++) {
      auto diff = abs(h_C[i * M + j] - C[i * M + j]);
      if (diff > abl) {
        printf("The difference in %d row and %d col is %f which is larger than "
               "the abl:%f \n",
               i, j, diff, abl);
        return false;
      }
    }
  }
  // free the buffer
  free(h_C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return true;
}

template <typename dtype> void init_matrix(dtype *mat, int size) {
  for (int i = 0; i < size; i++) {
    mat[i] = rand() / float(RAND_MAX);
  }
}

int main() {
  float *A, *B, *C;
  int M = 512, N = 512, K = 512;
  auto size_a = M * K * sizeof(float);
  auto size_b = K * N * sizeof(float);
  auto size_c = M * N * sizeof(float);

  A = (float *)malloc(size_a);
  B = (float *)malloc(size_b);
  C = (float *)malloc(size_c);

  srand(time(0));
  init_matrix(A, M * K);
  init_matrix(B, K * N);
  auto res = test_acc(A, B, C, M, N, K);
  if (res)
    printf("Accuracy test pass");
  free(A);
  free(B);
  free(C);
}
