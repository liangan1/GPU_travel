## 1. 为什么大规模并行编程会成为AI时代的标配？
需求与供给的角度来解释
### 硬件的角度
- 如何简单计算硬件的算力
- 统计里面来 CPU 和 GPU 频率的变化
- 解释当前硬件提高算力的方式
 - CPU
    -  SIMD
    -  多核
 - GPU
   - SM/CUDA core 数量
   - Tensor Core
### 统计当前大模型算力的需求

### 多维并行
 - 分布式并行
 - 多线程并行
 - 指令并行

### AI 时代的内存墙及应对策略？


## 2. GPU kernel 的 block 大小为什么要这么设置？
### 目标
GPU 是面向吞吐设计的。拥有更多的活跃 warp 可以隐藏那些
### 限制因素

### 如何计算 occupancy
### 参考文献   

## 3. 如何从并行编程的角度理解CPU和GPU的相同之处？
### 不同的并行编程模型
- SPMD
- SIMD
- SIMT 
- MIMD
- 
SMID lane 其实就是一个 warp lane， 以 AVX512 为例，一条指令有 16个 SIMD lane, 每个 SIMD lane处理 32bits的数据， 而 GPU 中， 每个warp有32个threads，即32个warp lane， 每个lane也是处理32bits的数据（1 个寄存器的宽度）。 那么对于 tensor core和 AMX 应该如何理解呢？ 在 GPU 中，wmma是一个warp function，即warp中的所有线程合作完成一条warp指令，这条warp指令在volta 架构中可以处理 16*16*16 的数据。
