#include<stdio.h>

__device__ constexpr int next_pow(unsigned int num){
	if(num <=1) return num;
	return 1 <<(CHAR_BIT *sizeof(num) - __builtin_clz(num-1));
}

template<typename T, int width=32>
__device__ void warp_reduce(T& val) {
  auto mask = __ballot_sync(0xffffffff, threadIdx.x <width);
  for(int lanemask = width>>1; lanemask>=1; lanemask=lanemask>>1){
     val += __shfl_xor_sync(mask, val, lanemask, width);
  }
}

template<typename T, int max_block_size=1024>
__device__ void block_reduce(T& val) {
     //only supprt a two-level hieracy warp reduce 
     static_assert(max_block_size <= 1024, "The number of thread in a block should less than 1024 to use the block_reduce");
     //use the warp reduce to get the partial sum of every warp
     warp_reduce(val);
     //the activate lanes 
     constexpr unsigned int max_active_lanes = (max_block_size + 32 - 1)/32;
     //store the result of the first lane in every warp into the shared memory 
     __shared__ T psum[max_active_lanes];
     auto laneid = threadIdx.x % 32;
     auto warpid = threadIdx.x / 32;
     if(laneid == 0) {
	psum[warpid] = val;
     }
     //use the warp reduce to get the final sum 
     val = threadIdx.x < blockDim.x/32 ? psum[threadIdx.x]:0;
     warp_reduce<T, next_pow(max_block_size/32)>(val);  
     
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
   block_reduce<T>(tsum);
   printf("\n after warp reduce tsum:%d", tsum);
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
   constexpr int len = 32*32;
   int array[len]={0};
   for(int i = 0; i < len; i++){
	   array[i]=32+ i%64;
   }
   auto res_cpu = sum<int>(array, len);
  
   int* d_array;
   cudaMalloc(&d_array, len * sizeof(int));
   cudaMemcpy(d_array, array, len*sizeof(int), cudaMemcpyHostToDevice);
   sum_cuda<int><<<1, 32>>>(d_array, len);
   cudaMemcpy(array, d_array, len * sizeof(int), cudaMemcpyDeviceToHost);
   printf("\nres_cuda:%d \n", array[0]);
   printf("res_cpu:%d \n", res_cpu);
}


