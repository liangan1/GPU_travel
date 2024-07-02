## Nvidia White Paper 
### [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.cn/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- First Generation Tensor Core
- New combined L1 data cache and shared memory unit
- 16 GB HBM2 memory subsystem delivers 900 GB/sec peak memory 
bandwidth.
- Second-Generation NVIDIA NVLink™ , six NVLink links and total bandwidth of 300 GB/sec
- The tensor core process 4x4x4 MM.  At the CUDA level, the warp-level interface assumes 16x16 size matrices spanning all 32 threads of the warp? how these data mapped to the tensor core and thread?
  

