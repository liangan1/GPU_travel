# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward

from flash_attn import flash_attn_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


def flops(batch, seqlen_q, seqlen_kv, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen_q * seqlen_kv * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # Adding is faster than masked_fill_
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd(func, *args, **kwargs):
    _, m = benchmark_forward(func, *args, **kwargs)
    return m.mean


repeats = 100
device = 'cuda'
dtype = torch.float16

#warmup
q = torch.randn(32, 4096, 32, 128, device=device, dtype=dtype, requires_grad=True)
k = torch.randn(32, 4096, 32, 128, device=device, dtype=dtype, requires_grad=True)
v = torch.randn(32, 4096, 32, 128, device=device, dtype=dtype, requires_grad=True)
f = time_fwd(flash_attn_func, q, k, v, 0.0, causal=0,
             repeats=500, verbose=False
            )
from datetime import datetime
# get current time
now = datetime.now()

# format with milliseconds
formatted_time = now.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]

print(formatted_time)

bs_seqlen_vals = [(16, 1024, 1024), (4, 1024, 4096), (1, 4096, 4096)]
causal_vals = [False]
headdim_vals = [128]
kv_heads = [32, 8]
dim = 4096
dropout_p = 0.0

methods = ["Flash2"]

time_f = {}

for causal in causal_vals:
    for headdim in headdim_vals:
        for batch_size, seqlen_q, seqlen_kv in bs_seqlen_vals:
            nheads = dim // headdim
            for nheads_kv in kv_heads:
                config = (causal, headdim, batch_size, seqlen_q, seqlen_kv, nheads_kv)

                # FlashAttention 2 (q, k, v separate to allow seqlen_q != seqlen_kv and GQA)
                if "Flash2" in methods:
                    q = torch.randn(batch_size, seqlen_q, nheads, headdim,
                                    device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(batch_size, seqlen_kv, nheads_kv, headdim,
                                    device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(batch_size, seqlen_kv, nheads_kv, headdim,
                                    device=device, dtype=dtype, requires_grad=True)
                    f = time_fwd(
                        flash_attn_func, q, k, v, dropout_p, causal=causal,
                        repeats=repeats, verbose=False
                    )
                    time_f[config, "Flash2"] = f
                # get current time
                now = datetime.now()

                # format with milliseconds
                formatted_time = now.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]

                print(formatted_time)

                # Report
                print(
                    f"### causal={causal}, headdim={headdim}, batch_size={batch_size}, "
                    f"seqlen_q={seqlen_q}, seqlen_kv={seqlen_kv}, kv_heads={nheads_kv} ###"
                )
                for method in methods:
                    if (config, method) not in time_f:
                        continue
                    flop = flops(batch_size, seqlen_q, seqlen_kv, headdim, nheads, causal, mode="fwd")
                    time_ms = time_f[config, method] * 1e3
                    print(
                            f"{method} {efficiency(flop, time_f[config, method]):.2f} TFLOPs/s, time: {time_ms:.3f} ms"
                    )

# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
~
~
