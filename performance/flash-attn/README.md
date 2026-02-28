## Introduction
This file is used to describe the performance of Flash-Attention V2 on the SM80. The code base of CUDA is based on the https://github.com/Dao-AILab/flash-attention/tree/main

## Benchmark Result (TFLOPs/s & ms）
The peak frequency is 1410MHZ and it will drop to ~1200MHZ, so the peak TFLOPs of A100 should be ~265TFLOPs/s when runing the FAV2. 
For non-causal, when we disable the softmax in SDPA, the GEMM efficiency should be ~83%(220/265), while the efficiency drop to 68% w/ softmax.
For causal, the efficiency is 37%~60%. 
| bs | seqlen_q | seqlen_kv | kv_heads | Baseline TFLOPs/s | Baseline ms | Barrier TFLOPs/s | Barrier ms | No softmax TFLOPs/s | No softmax ms |
|---:|---------:|----------:|---------:|-----------------:|------------:|-----------------:|-----------:|--------------------:|--------------:|
| 16 | 1024 | 1024 | 32 | 177.67 | 1.547 | 178.30 | 1.542 | 217.51 | 1.264 |
| 16 | 1024 | 1024 | 8  | 180.91 | 1.519 | 179.43 | 1.532 | 218.34 | 1.259 |
| 4  | 1024 | 4096 | 32 | 183.27 | 1.500 | 182.51 | 1.506 | 217.41 | 1.264 |
| 4  | 1024 | 4096 | 8  | 185.22 | 1.484 | 185.02 | 1.486 | 220.61 | 1.246 |
| 1  | 4096 | 4096 | 32 | 185.64 | 1.481 | 186.94 | 1.470 | 219.35 | 1.253 |
| 1  | 4096 | 4096 | 8  | 187.13 | 1.469 | 186.89 | 1.471 | 223.02 | 1.233 |



| causal | bs | seqlen_q | seqlen_kv | kv_heads | TFLOPs/s | time (ms) |
|:------:|---:|---------:|----------:|---------:|---------:|----------:|
| True  | 16 | 1024 | 1024 | 32 | 143.70 | 0.956 |
| True  | 16 | 1024 | 1024 | 8  | 140.59 | 0.978 |
| True  | 4  | 1024 | 4096 | 32 | 98.10  | 1.401 |
| True  | 4  | 1024 | 4096 | 8  | 99.84  | 1.377 |
| True  | 1  | 4096 | 4096 | 32 | 156.27 | 0.879 |
| True  | 1  | 4096 | 4096 | 8  | 159.41 | 0.862 |
| False | 16 | 1024 | 1024 | 32 | 179.11 | 1.535 |
| False | 16 | 1024 | 1024 | 8  | 179.48 | 1.532 |
| False | 4  | 1024 | 4096 | 32 | 179.89 | 1.528 |
| False | 4  | 1024 | 4096 | 8  | 182.60 | 1.505 |
| False | 1  | 4096 | 4096 | 32 | 184.04 | 1.494 |
| False | 1  | 4096 | 4096 | 8  | 183.63 | 1.497 |
## Others 
### Frequency
- A100 frequency drop due to the power limit(peak 1410MZH --> ~1200 MHZ).
- Frequency monitor with nvidia-smi 
  
   nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active --format=csv -lms 10 2>&1|tee format.csv
