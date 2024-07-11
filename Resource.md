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
  - How about the programing model comapred to the V100, still process 16x16x16?
  - How to understand the folloiwng Fig.14 in the A100 white paper? It seems that if we can understand this table correctly we already underdant the SIMT mode and Tensor Core implementation.
    <img width="645" alt="image" src="https://github.com/liangan1/cuda_travel/assets/46986936/f5e17c1a-b7f2-4073-bbec-78ce620ba163">
   
   - Physical Tensor Core Configuration vs. Instructional Capability
   
   - **Physical Configuration (8x4x8)**: The physical layout of a Tensor Core in the A100 architecture might be described in terms of the dimensions it can process in a single operation. This configuration indicates the core's design to efficiently handle matrix operations of a certain size, which in this case, suggests an 8x4x8 operation.
   
   - **Instructional Capability (16x8x16)**: Despite the physical configuration, the A100 GPU introduces enhanced Tensor Core instructions that effectively utilize multiple Tensor Cores in tandem to perform larger matrix operations, such as 16x8x16. This enhancement is a result of architectural improvements that allow for more flexible and efficient use of Tensor Cores, enabling them to operate on larger blocks of data than what might be suggested by their physical layout alone.
   
   ### Why the Enhanced Instructions?
   
   - **Increased Flexibility and Efficiency**: By supporting larger matrix operations through enhanced instructions, the A100 can more efficiently handle a wider range of workloads, particularly those requiring larger matrix sizes, without needing to break them down into smaller chunks.
   
   - **Optimized Performance**: The enhanced instructions are designed to optimize the performance of deep learning and HPC (High-Performance Computing) applications by maximizing the utilization of the Tensor Cores' computational capabilities.
   
   - **Software Abstraction**: From a software development perspective, these enhanced instructions provide a more abstracted and simplified interface for leveraging the power of Tensor Cores, making it easier for developers to write optimized code without needing to manage the complexities of the underlying hardware.

  


## ISPC history 
[The story of ispc: all the links](https://pharr.org/matt/blog/2018/04/30/ispc-all)

