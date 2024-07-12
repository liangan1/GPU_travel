## Nvidia White Paper 

### Architecture evoluation for NV GPUs
ToDO: provide the tables to show these difference. 
- SMs
   - CUDA Core
   - Tensor Core
- Memory
  - On-chip memory 
    - Shared memory
    - L1
    - L2
  - Off-chip memory 
    - HBM
- NVLinks

### [NVIDIA V100 GPU Architecture](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- First Generation Tensor Core
  - Only FP16 is supported to do F16/FP32 mixed FMA. Add INT8/INT4/INT1 supported in the turing architecture.
  - 2 tensor core partition per SM and 4 tensor core per partition.8 tensor cores per SM. Every tensor core compute 4x4x4 FMAs per cocloc which means 512 FMAs can be finished per colock for every SM.
  - In the programing level, the warp matrix function calculate 16x16x16 FMAs. A GEMM example can be found [Get started with Tensor Cores in CUDA 9 today](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
    
- New combined L1 data cache and shared memory unit
- 16 GB HBM2 memory subsystem delivers 900 GB/sec peak memory 
bandwidth.
- Second-Generation NVIDIA NVLink™ , six NVLink links and total bandwidth of 300 GB/sec
- The tensor core process 4x4x4 MM.  At the CUDA level, the warp-level interface assumes 16x16 size matrices spanning all 32 threads of the warp? how these data mapped to the tensor core and thread?
  
### [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- Third gerneration Tensor Core
  - Add BF16, FP32, TF32, FP64 compared to Turing Achitecture and supported FP16 accumulate
    <img width="679" alt="image" src="https://github.com/liangan1/cuda_travel/assets/46986936/d6d41d6c-f95c-4ad4-b36b-34bb52395fe3">
  - Acceleration for all data types including FP16, BF16, TF32, FP64, INT8, INT4 and Binary
    <img width="679" alt="image" src="https://github.com/liangan1/cuda_travel/assets/46986936/0ba573f1-6b6e-40ad-9827-a72fb167d906">
  - How about the programing model comapred to the V100, still process 16x16x16? How to understand the folloiwng Fig.14 in the A100 white paper? It seems that if we can understand this table correctly we already underdant the SIMT mode and Tensor Core implementation.
    
    There are 3-level hieracy to understand the tensor core implementation: warp level CUDA function/PTX instrinsic, SASS ISA instrinsic and hardware tensor core MMA.  For the programmer, A warp lelvel function nvcuda::wmma can be used to collaberatively process a specify shape matrix multiplication and accumulation D=A*B + C. All threads in this warp collaberately load the fragment of A/B and store the result to the D from C after wmma. The PTX[2] also provide a same level intrinsics to do this operation. The supported shape for every data type can found in [wmma:warp level matrix shape](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#warp-level-matrix-shape) The PTX instrinsic may be mapped to multiple SASS ISA intrinsics and A SASS ISA instrinsic is also translated multiple hard ware tensor cores. Let us take warp-level shape MxNxK: 16x16x16 and BF16 input data type as a example. A PTX intrinsic is mapped to 2 16x8x16 intrincs. We know that every tensor core only process 8x4x8/cycle, so A SASS 16x8x16 instrinsic is translated to 8 hardware MMA instrunctions and need 8 cycles. For the operands, A framents include 16x16/32*2 bytes and need 4 b32 register to store it, similar to B. For the C and D, the acculator data type fo BF16 MMA is the FP32, so 8 register are used for each of them. The distribution of fragments loaded by the threads in a warp is unspecified and is target architecture dependent.
    
    
    <img width="645" alt="image" src="https://github.com/liangan1/cuda_travel/assets/46986936/f5e17c1a-b7f2-4073-bbec-78ce620ba163">
    <img width="598" alt="image" src="https://github.com/user-attachments/assets/fa84be13-de4f-49df-9889-5ff5dfb14f3b">
    
   
    - Reference to understand the tensor core implementation.
      - [1] [PTX: warp lelvel matrix multiply accumulate instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=wmma#warp-level-matrix-multiply-accumulate-instructions)
      - [2] [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/pdf/2208.11174) 
   
   

  


## ISPC history 
[The story of ispc: all the links](https://pharr.org/matt/blog/2018/04/30/ispc-all)

