# cuda_travel
A project to record the cuda learning process

# Leanrning ROADMAP 
|   TimeLine |Task  |Goal| Status
| -----------| ------------- |-----------| ------------- |
|2024.Q2     |CUDA Programing Guide &CUDA-GDB| 1. Key concept of  CUDA(Runtime, Hardware Architecture, Performance Guideline) 2.CUDA-GDB for matmul kernel |   |   |
|2024.Q3     |LLM kernels & NVSight | 1.Understand and program for layerNorm, Softmax & PagedAttention 2.Familiar with NVSight to breakdown the HW resources analysis 
|2024.Q4     | TensorRT & Vllm  based on CUDA|1. Familiar with the usage of tensorRT and VLLM based on A100-80G  2.Analysis the HW utilization of Llama based on the above inference engine

# ToDo List
	1. How to use the nvtx?
	2. Why need cuda synchronize? When and how to use ?
	3. How to use cuda stream for pytorch?
	4. How to understand the memory management in pytorch? 
	
	5. How to understand the cuda profile?
		a. Tensorboard profiler
		b. Self cpu total should close to the self cuda total.
		c. CudaKernel launch is high?
			i. Graph fusion 
				1) How to apply op fusion with CUDA?
				2) What is the relationship between the cuda graph and jit graph/dynamo fx graph?
			ii. CudaGraph in vllm ->Done
			
			
				1) Store multiple cuda graph with different batch size, need to copy the input to the trace buffer. Need more memory usage with cuda graph. 
				2) Cuda graph is applied to only the next token(decoding stage), no cuda graph for the first token(prefill stage). Only model.forward in cuda graph, no cuda graph for other part, e.g., llm head and sample(beam. Greedy).  ->Done
				3) Performance compare w/ and w/o cuda graph using Vllm ->Done. 
				4) Cuda graph for pytorch+huggingface with pagedattention. 
		d.  Bank conflict 
			i. How to detect the bank conflict with nsys? 
				1) Is there any bank conflict for the vllm memory bound operators? 
						i) Rmsnorm
						ii) rope
						iii) paged attention
				2) How to furtherly improve the memory bw utilization for these above ops?
					
		e. Occupancy 
			i. What is the occupancy? What does it means? 
			Achieved Occupancy
			<https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm> 
			
			
			The SM has the maximum active warps that can be active at once. The occupancy is the ratio of active warps to the maximums supported active warps. 
			
			Issue Efficiency
			https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/issueefficiency.htm#WarpIssueEfficiency
			Device Limit >= Theoretical Occupancy >= Active Warps
			
			
			The set of active warps can be thought of as the pool of candidates for making forward progress at runtime at a given moment in time. Stalled Warps not able to make forward progress and the sub-set of Eligible Warps that are ready to issue their next instruction.
			
				Active Warps == Stalled Warps+ Eligible Warps 
			
			
			
			
			ii. How to understand the occupancy goal for vllm?
			iii. How to measure there are enough occupancy can hidden the latency? How to select the  best occupancy configuration on a given device? Is there any tools to see it?
		f. How to debug the CUDA kernel?
		g. 如何计算一张卡的理论算力？
		h. 如何测试GPU的实际带宽？类似MLC那种工具
		i. GPU memory transaction?
		https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/sourcelevel/memorytransactions.htm
		
			
	6. By end of 6.10 
		a. Tensor core evolution?
		
		https://www.zhihu.com/question/654521367
		Using CUDA Graph in Pytorch
		https://medium.com/@yuanzhedong/using-cuda-graph-in-pytorch-80898b395a38
		
		b. CUDA graph 
			i. What is cuda graph? --> Done 
			https://blog.fireworks.ai/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning-353bf6241248
			ii. Where to disable cuda graph in vllm? --> Done 
				1) --enforce_eager  -> ~1.5x performance gap for 1024in/128out with prompt=10 on A100
			iii. How to enable cuda graph in autoAWQ?
		c. Achieved occupancy 
			i. How to understand the occupancy?
			ii. How to understand the information in the nsight? 





