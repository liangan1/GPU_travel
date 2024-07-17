#include<stdio.h>

template<typename T>
__device__ void warp_reduce(T& val) {
  for(int i = 16; i >=1; i=i>>1){
     val += __shfl_xor_sync(0xffffffff, val, i, 32);
  }
}

template<typename T>
__global__ void sum_cuda(T* array, int len){
   auto idx = blockIdx.x * blockDim.x + threadIdx.x;
   T tsum =0;
   for(int i = idx; i < len; i+=blockDim.x)
   {
      tsum += array[i];
      printf("%d ", tsum);
   }
   warp_reduce<T>(tsum);
   //printf("\n after warp reduce tsum:%d", tsum);
   if(idx==0){
     array[0] = tsum;
   }

}

template<typename T>
T sum(T *array, int len){
   T sum = 0;
   for(int i = 0; i < len; i++){
      sum += array[i]; 	   
   }
   return sum;
}



int main(){
   int len = 32*4;
   int array[32*4]={0};
   for(int i = 0; i < len; i++){
	   array[i]=i%64;
   }
   auto res_cpu = sum<int>(array, len);
  
   int* d_array;
   int res_cuda = 0;
   cudaMalloc(&d_array, len * sizeof(int));
   cudaMemcpy(d_array, array, len*sizeof(int), cudaMemcpyHostToDevice);
   sum_cuda<int><<<1, 32>>>(d_array, len);
   cudaMemcpy(array, d_array, len * sizeof(int), cudaMemcpyDeviceToHost);
   printf("\nres_cuda:%d \n", array[0]);
   printf("res_cpu:%d \n", res_cpu);
}


