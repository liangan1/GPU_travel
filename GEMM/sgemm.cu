#include<iostream>

#define OFFSET(m, n, lda) m*lda + n 
/*
   * C = A @ B 
   * A,B,C are row-major matrix 
   * A:(M,K), B:(K,N), C(M,N)
   *
*/
__global__ sgemm(float* A, float*B, float* C, int M, int N, int K)
{
    //thread info 
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    
    //building block info
    // 1) A thread block calculate a 128*128 sub-matrix in the C matrix 
    // 2) A thread calculate a 8*8 sub-matrix in the C matrix 
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    float c_r[TM][TN]={0};

    //The element used to calculate the 128*128 sub-matrix should be stored in the SHM 
    __shared__ float* s_a[BM][BK];
    __shared__ float* s_b[BN][BK];

    //Load elements from global memory to the shared memory 
    /*
     *1) Elements number loaded for every thread: 
     *   There BM/TM * BN/TN threads and BM*BK elements in A sub-matrix and BK*BN elemens in the B sub-matrix 
     *   So, every thread load BK*TM*TN/BN=4 elements from A and BK*TM*TN/BM=4 elements from the B  
     *2) There are 8 elemnts in every row A sub-matrix to load and 128 elments in every row of sub-matrix B 
     *   So, Float4 can be used to speed-up the load 
     *
     *
     */
     //1) local row/col id for every thread 
     int shm_a_load_row = tid >> 1;//tid/2, row id in the s_a, 2 Float4 in every row
     int shm_a_load_col = tid & 1; //tid%2, col id in the s_a 
     int shm_b_load_row = tid >>5; //tid/32, row id in the s_b, 128/4=32 Float4 in every row
     int shm_b_load_col = (tid%32) / 4;
     
     //2) global row/col id for every thread, need to add the block offset 

     int gemm_a_load_row = by * BM + shm_a_load_row;
     int gemm_b_load_col = bx * BN + shm_b_load_col;

     for(int bk = 0; bk < (K+BK-1)/BK; bk+=BK){
	 int gemm_a_load_col = bk * BK + shm_a_load_col;
         int gemm_a_load_row = bk * BK + shm_b_load_row;
         float4*(s_a[shm_a_load_row][shm_a_load_col])=float4*(A[OFFSET(gemm_a_load_row, gemm_a_load_col, K)]);
	 float4*(s_b[shm_b_load_row][shm_b_load_col])=float4*(A[OFFSET(gemm_b_load_row, gemm_b_load_col, N)]);
	 __synchronize();
	 //calculate the C(TM, TN)
	 for(int k = 0; k < BK; k++){
		 for(int i = 0; i < TM; i++){
			 for(int j = 0; j < TN; j++){
				 c_r[i][j] += s_a[shm_a_load_row][k] * s_b[k][shm_b_load_col]; 
			 }
		 }
	 }
     }
     
     //store the c_r result into the global memory     
    int gemm_c_store_row = by * BM + ty * TM;
    int gemm_c_store_col = bx * BM + tx * TN;
    for(int i =0 ; i < TM; i++){
	    for(int j = 0; j < TN; j++){
		    C[OFFSET(gemm_c_store_row+i, gemm_c_store_col+j, N)] = c_r[i][j];
	    }
    }
}

int main()
{
	
}
