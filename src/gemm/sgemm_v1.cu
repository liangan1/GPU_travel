#include <float.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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
  __shared__ float s_a[2][BK][BM];
  __shared__ float s_b[2][BK][BN];

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
  int shm_a_load_m =
      tid >> 1; // tid/2, row id in the s_a, 2 Float4 in every row
  int shm_a_load_k = (tid & 1) << 2; // tid%2, 0 OR 4, col id in the s_a
  int shm_b_load_k =
      tid >> 5; // tid/32, row id in the s_b, 128/4=32 Float4 in every row
  int shm_b_load_n = (tid & 31) << 2;

  float r_a[4];
  float r_c_a[8];
  float r_c_b[8];
  // 2) global row/col id for every thread, need to add the block offset

  int gemm_a_load_m = by * BM + shm_a_load_m;
  int gemm_b_load_n = bx * BN + shm_b_load_n;
  // load the bk=0
  int bk = 0;

  int gemm_a_load_k = shm_a_load_k;
  auto gemm_a_load_offset = OFFSET(gemm_a_load_m, gemm_a_load_k, K);
  FLOAT4(r_a) = FLOAT4(&(A[gemm_a_load_offset]));
  s_a[0][shm_a_load_k][shm_a_load_m] = r_a[0];
  s_a[0][shm_a_load_k + 1][shm_a_load_m] = r_a[1];
  s_a[0][shm_a_load_k + 2][shm_a_load_m] = r_a[2];
  s_a[0][shm_a_load_k + 3][shm_a_load_m] = r_a[3];
  int gemm_b_load_k = shm_b_load_k;
  auto gemm_b_load_offset = OFFSET(gemm_b_load_k, gemm_b_load_n, N);
  FLOAT4(&(s_b[0][shm_b_load_k][shm_b_load_n])) =
      FLOAT4(&(B[gemm_b_load_offset]));
  __syncthreads();

  int shm_index = 0;
  for (bk = 1; bk < (K + BK - 1) / BK; bk++) {
    shm_index = (bk - 1) & 1;
    // calculate the C(TM, TN)
    for (int k = 0; k < BK; k++) {
      FLOAT4(&r_c_a[0]) = FLOAT4(&s_a[shm_index][k][ty * TM / 2]);
      FLOAT4(&r_c_a[4]) = FLOAT4(&s_a[shm_index][k][ty * TM / 2 + BM / 2]);
      FLOAT4(&r_c_b[0]) = FLOAT4(&s_b[shm_index][k][tx * TN / 2]);
      FLOAT4(&r_c_b[4]) = FLOAT4(&s_b[shm_index][k][tx * TN / 2 + BN / 2]);
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          c_r[i][j] += r_c_a[i] * r_c_b[j];
        }
      }
    }
    int shm_load_index = bk & 1;

    int gemm_a_load_k = bk * BK + shm_a_load_k;
    auto gemm_a_load_offset = OFFSET(gemm_a_load_m, gemm_a_load_k, K);
    FLOAT4(r_a) = FLOAT4(&(A[gemm_a_load_offset]));
    s_a[shm_load_index][shm_a_load_k][shm_a_load_m] = r_a[0];
    s_a[shm_load_index][shm_a_load_k + 1][shm_a_load_m] = r_a[1];
    s_a[shm_load_index][shm_a_load_k + 2][shm_a_load_m] = r_a[2];
    s_a[shm_load_index][shm_a_load_k + 3][shm_a_load_m] = r_a[3];
    int gemm_b_load_k = bk * BK + shm_b_load_k;
    auto gemm_b_load_offset = OFFSET(gemm_b_load_k, gemm_b_load_n, N);
    FLOAT4(&(s_b[shm_load_index][shm_b_load_k][shm_b_load_n])) =
        FLOAT4(&(B[gemm_b_load_offset]));
    __syncthreads();
  }

  // calculate the last bk
  shm_index = (bk - 1) & 1;
  for (int k = 0; k < BK; k++) {
    FLOAT4(&r_c_a[0]) = FLOAT4(&s_a[shm_index][k][ty * TM / 2]);
    FLOAT4(&r_c_a[4]) = FLOAT4(&s_a[shm_index][k][ty * TM / 2 + BM / 2]);
    FLOAT4(&r_c_b[0]) = FLOAT4(&s_b[shm_index][k][tx * TN / 2]);
    FLOAT4(&r_c_b[4]) = FLOAT4(&s_b[shm_index][k][tx * TN / 2 + BN / 2]);
    for (int i = 0; i < TM; i++) {
      for (int j = 0; j < TN; j++) {
        c_r[i][j] += r_c_a[i] * r_c_b[j];
      }
    }
  }
  __syncthreads();

  // store the c_r result into the global memory
  for (int i = 0; i < TM / 2; i++) {
    int gemm_c_store_m = by * BM + ty * TM / 2 + i;
    int gemm_c_store_n = bx * BN + tx * TN / 2;
    int gemm_c_store_addr = OFFSET(gemm_c_store_m, gemm_c_store_n, N);
    FLOAT4(&C[gemm_c_store_addr]) = FLOAT4(&c_r[i][0]);
    FLOAT4(&C[gemm_c_store_addr + BN / 2]) = FLOAT4(&c_r[i][4]);
  }
  for (int i = 0; i < TM / 2; i++) {
    int gemm_c_store_m = by * BM + ty * TM / 2 + BM / 2 + i;
    int gemm_c_store_n = bx * BN + tx * TN / 2;
    int gemm_c_store_addr = OFFSET(gemm_c_store_m, gemm_c_store_n, N);
    FLOAT4(&C[gemm_c_store_addr]) = FLOAT4(&c_r[i + TM / 2][0]);
    FLOAT4(&C[gemm_c_store_addr + BN / 2]) = FLOAT4(&c_r[i + TM / 2][4]);
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

float testPerformance(void (*gpuSgemm)(float *, float *, float *, const int,
                                       const int, const int),
                      dim3 gridDim, dim3 blockDim, const int M, const int N,
                      const int K, const int repeat) {

  size_t size_a = M * K * sizeof(float);
  size_t size_b = K * N * sizeof(float);
  size_t size_c = M * N * sizeof(float);

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size_a);
  cudaMalloc(&d_b, size_b);
  cudaMalloc(&d_c, size_c);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++)
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float msec, sec;
  cudaEventElapsedTime(&msec, start, end);
  sec = msec / 1000.0 / repeat;

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return sec;
}

void test_performance() {
  printf("\nKernal = sgemm_V1\n");
  const int outer_repeat = 1, inner_repeat = 1;
  const int BM = 128, BN = 128, TM = 8, TN = 8;
  void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) =
      sgemm_v1;

  const int M_list[15] = {128,  192,  256,  384,  512,  768,   1024, 1536,
                          2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int N_list[15] = {128,  192,  256,  384,  512,  768,   1024, 1536,
                          2048, 3072, 4096, 6144, 8192, 12288, 16384};
  const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
                          1024, 1024, 1024, 1024, 1024, 1024, 1024};

  const int TESTNUM = 15;
  for (int i = 0; i < TESTNUM; i++) {
    const int M = M_list[i], N = N_list[i], K = K_list[i];

    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;

    for (int j = 0; j < outer_repeat; j++) {
      double this_sec =
          testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
      max_sec = max(max_sec, this_sec);
      min_sec = min(min_sec, this_sec);
      total_sec += this_sec;
    }

    double avg_sec = total_sec / outer_repeat;
    double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

    printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG "
           "Performance = %10.4lf Gflops\n",
           M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
  }
}

int main() {
  cudaDeviceProp prop;
  int device;

  // Get the default CUDA device
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  // Print the available shared memory per block (SM)
  std::cout << "Available shared memory per block: " << prop.sharedMemPerBlock
            << " bytes" << std::endl;
  // Print the number of 32-bit registers available per block
  std::cout << "Available 32-bit registers per block: " << prop.regsPerBlock
            << std::endl;
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
    printf("Accuracy test pass \n");
  test_performance();
  free(A);
  free(B);
  free(C);
}
