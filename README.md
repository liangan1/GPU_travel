# cuda_travel
A project to record the cuda learning process

# Leanrning ROADMAP 
|   TimeLine |Task  |Goal| Status
| -----------| ------------- |-----------| ------------- |
|2024.Q2     |CUDA Programing Guide &CUDA-GDB| 1. Key concept of  CUDA(Runtime, Hardware Architecture, Performance Guideline) 2.CUDA-GDB for matmul kernel | Done |
|2024.Q3     |LLM kernels & NVSight | 1.Understand and program for layerNorm, Softmax & PagedAttention 2.Familiar with NVSight to breakdown the HW resources analysis 
|2024.Q4     | TensorRT & Vllm  based on CUDA|1. Familiar with the usage of tensorRT and VLLM based on A100-80G  2.Analysis the HW utilization of Llama based on the above inference engine

## What I have done?
### GEMM
- CUDA core FP32 SGEMM kernel
  - [#PR2](https://github.com/liangan1/cuda_travel/pull/2)
    - 17.1/19.5=87.7% FLOPS ultilization on A100-80G
    - Improve the compute memory access ratio with tiling and stripe mining using shared memory 
    - Resolve bank conflict
    - Double buffer 
    - Occupancy Analysis
### Reduction 
- Warp reduce and Block reduce
  - [#PR3](https://github.com/liangan1/cuda_travel/pull/3)
      - Warp reduce with warp shuffle instrinsics 
      - Blokc reduce with 2-level hierarcy warp reduce and shared memory

## Working in Progress 
- RMSNorm Kernel
- Softmax Kernel
  
# ToDo List
## How to Learn the details of your GPU HW?
- HW details
  - nvidia-smi: [nvidia-smi: Control Your GPUs](https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/)
  - Compute Capabity
    - nvidia-smi --query-gpu=compute-cap
    - queryDevice([cuda-sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery))
  - Memory Bandwidth
    - queryDevice([cuda-sample](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities/deviceQuery))
  - Memory Capacity 
  - FLOPS
    - Calculate the FLOPS according to the frequence, SM, core info.
       - CUDA Core FLOPS
         -  FLOPS = #CUDA_CORE * Frequency * 2(FMA)
            - A100-80G: 6912 * 1410M * 2 = 19.49 TFLOPS [Ampher Wiki](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)#:~:text=The%20A100%20features%2019.5%20teraflops,s%20of%20graphics%20memory%20bandwidth.)
       - Tensor Core FLOPS
  
## Debug

### cuda-gdb 

## OpenAI Triton 

## Profile 

### Nsight Compute 

