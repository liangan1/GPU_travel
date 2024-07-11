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
  


## Roofline model 
https://www.nersc.gov/assets/Uploads/Tutorial-ISC2018-Roofline-Model.pdf

## ISPC history 
[The story of ispc: all the links](https://pharr.org/matt/blog/2018/04/30/ispc-all)

