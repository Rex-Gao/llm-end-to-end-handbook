# 05 · Triton与多卡

[阅读说明](../README.md) · [完整手册](../大模型全链路知识手册.md) · [上一册：资源推理与测量](04_资源推理与测量.md) · [下一册：Scaling与评测](06_Scaling与评测.md)

<a id="chapter-10"></a>
## 10. 手写 Triton Attention 内核

### 10.1 实验目标与支持范围

实现一个数学上等价于 causal scaled dot-product Attention 的 tiled kernel：不把 `T×T` 权重矩阵写入 HBM，保存每行 logsumexp，backward 重建局部概率。FlashAttention 的关键是 IO-aware tiling 与重算，不是近似删掉注意力项。[R16](12_模板公式与参考.md#r16)、[R17](12_模板公式与参考.md#r17)

本章代码是教学实现，支持 contiguous `[B,H,T,D]`、相同 Q/K/V shape、`D∈{64,128}`、FP16／BF16、causal、无 dropout、无 padding／GQA，不支持高阶梯度。它没有实现完整 FlashAttention-2 的调度与优化，不保证比 SDPA 快。代码须在支持的 Linux CUDA／Triton 组合上完成编译、前向／反向误差检查与性能测量，按第 10.6 节记录结果。[R19](12_模板公式与参考.md#r19) 初学 Triton 可先写 vector add 和 fused softmax，再实现 Attention。[R18](12_模板公式与参考.md#r18)

### 10.2 Online softmax 推导

处理一行 attention scores 的若干块，维护最大值 m、归一化分母 l、未归一化输出 accumulator a。新块 S 到来：

$$
\begin{aligned}
m'&=\max(m,\max S),\\
\alpha&=e^{m-m'},\quad P=e^{S-m'},\\
l'&=\alpha l+\sum P,\\
a'&=\alpha a+PV.
\end{aligned}
$$

最后 `O=a/l`，保存 `LSE=m+log(l)`。所有 accumulator 用 FP32。块边界不改变 softmax 的数学结果，但低精度转换和求和顺序会造成数值误差。

如果用 `exp2` 加速，则 score 要乘 `log2(e)`，保存的 logsumexp 也要说明是 log2 还是自然 log。本章统一 `exp`／自然 log，避免 forward 与 backward 混用底数。

### 10.3 Backward 推导

对一张 head：

$$
S=QK^\top/\sqrt D,\quad P=\operatorname{softmax}(S),\quad O=PV.
$$

令上游梯度为 `dO`：

$$
\begin{aligned}
dV&=P^\top dO,\\
dP&=dOV^\top,\\
\Delta_i&=\sum_jP_{ij}dP_{ij}=\sum_cO_{ic}dO_{ic},\\
dS&=P\odot(dP-\Delta[:,None]),\\
dQ&=dSK/\sqrt D,\quad dK=dS^\top Q/\sqrt D.
\end{aligned}
$$

局部 `P=exp(S-LSE[:,None])`，masked entries 设 0。按照 query block 分工算 dQ，按照 key block 分工算 dK／dV，可以避免多个 program 对同一输出的原子累加。本章分两个 backward kernel，重复计算一些局部 scores，以换取更直观的实现。

### 10.4 Forward 与 backward 教学代码

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(Q, K, V, O, LSE,
              N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    qb, bh = tl.program_id(0), tl.program_id(1)
    rm = qb * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    base = bh * N * D
    q = tl.load(Q + base + rm[:, None] * D + rd[None, :],
                mask=rm[:, None] < N, other=0)
    m = tl.full((BM,), -float("inf"), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for begin in range(0, N, BN):
        rn = begin + tl.arange(0, BN)
        kt = tl.load(K + base + rd[:, None] + rn[None, :] * D,
                     mask=rn[None, :] < N, other=0)
        v = tl.load(V + base + rn[:, None] * D + rd[None, :],
                    mask=rn[:, None] < N, other=0)
        scores = tl.dot(q, kt) * SCALE
        allowed = (rn[None, :] < N) & (rn[None, :] <= rm[:, None])
        scores = tl.where(allowed, scores, -float("inf"))
        new_m = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        den = alpha * den + tl.sum(p, axis=1)
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v)
        m = new_m
    result = acc / den[:, None]
    tl.store(O + base + rm[:, None] * D + rd[None, :],
             result, mask=rm[:, None] < N)
    tl.store(LSE + bh * N + rm, m + tl.log(den), mask=rm < N)


@triton.jit
def _attn_dq(Q, K, V, DO, LSE, DELTA, DQ,
             N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    qb, bh = tl.program_id(0), tl.program_id(1)
    rm, rd = qb * BM + tl.arange(0, BM), tl.arange(0, D)
    base = bh * N * D
    ptr = base + rm[:, None] * D + rd[None, :]
    q = tl.load(Q + ptr, mask=rm[:, None] < N, other=0)
    do = tl.load(DO + ptr, mask=rm[:, None] < N, other=0)
    lse = tl.load(LSE + bh * N + rm, mask=rm < N, other=0)
    delta = tl.load(DELTA + bh * N + rm, mask=rm < N, other=0)
    dq = tl.zeros((BM, D), tl.float32)
    for begin in range(0, N, BN):
        rn = begin + tl.arange(0, BN)
        kt = tl.load(K + base + rd[:, None] + rn[None, :] * D,
                     mask=rn[None, :] < N, other=0)
        vt = tl.load(V + base + rd[:, None] + rn[None, :] * D,
                     mask=rn[None, :] < N, other=0)
        scores = tl.dot(q, kt) * SCALE
        allowed = ((rm[:, None] < N) & (rn[None, :] < N)
                   & (rn[None, :] <= rm[:, None]))
        p = tl.where(allowed, tl.exp(scores - lse[:, None]), 0.0)
        dp = tl.dot(do, vt)
        ds = p * (dp - delta[:, None]) * SCALE
        dq += tl.dot(ds.to(kt.dtype), tl.trans(kt))
    tl.store(DQ + ptr, dq, mask=rm[:, None] < N)


@triton.jit
def _attn_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV,
               N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    kb, bh = tl.program_id(0), tl.program_id(1)
    rn, rd = kb * BN + tl.arange(0, BN), tl.arange(0, D)
    base = bh * N * D
    kp = base + rn[:, None] * D + rd[None, :]
    k = tl.load(K + kp, mask=rn[:, None] < N, other=0)
    v = tl.load(V + kp, mask=rn[:, None] < N, other=0)
    dk, dv = tl.zeros((BN, D), tl.float32), tl.zeros((BN, D), tl.float32)
    for begin in range(0, N, BM):
        rm = begin + tl.arange(0, BM)
        qp = base + rm[:, None] * D + rd[None, :]
        q = tl.load(Q + qp, mask=rm[:, None] < N, other=0)
        do = tl.load(DO + qp, mask=rm[:, None] < N, other=0)
        lse = tl.load(LSE + bh * N + rm, mask=rm < N, other=0)
        delta = tl.load(DELTA + bh * N + rm, mask=rm < N, other=0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        allowed = ((rm[:, None] < N) & (rn[None, :] < N)
                   & (rn[None, :] <= rm[:, None]))
        p = tl.where(allowed, tl.exp(scores - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * SCALE
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
    tl.store(DK + kp, dk, mask=rn[:, None] < N)
    tl.store(DV + kp, dv, mask=rn[:, None] < N)


class TritonCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("expected equal [B,H,T,D] tensors")
        if not q.is_cuda or not (q.device == k.device == v.device):
            raise ValueError("expected one CUDA device")
        if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
            raise ValueError("expected equal FP16/BF16 dtype")
        b, h, n, d = q.shape
        if b < 1 or h < 1 or n < 1 or d not in (64, 128):
            raise ValueError("unsupported dimensions")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out = torch.empty_like(q)
        lse = torch.empty((b, h, n), dtype=torch.float32, device=q.device)
        scale = 1 / math.sqrt(d)
        grid = (triton.cdiv(n, 32), b * h)
        _attn_fwd[grid](q, k, v, out, lse, n, d, scale, 32, 32, num_warps=4)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors
        do = do.contiguous()
        b, h, n, d = q.shape
        delta = (out.float() * do.float()).sum(-1).contiguous()
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        grid = (triton.cdiv(n, 32), b * h)
        _attn_dq[grid](q, k, v, do, lse, delta, dq,
                       n, d, ctx.scale, 32, 32, num_warps=4)
        _attn_dkdv[grid](q, k, v, do, lse, delta, dk, dv,
                         n, d, ctx.scale, 32, 32, num_warps=4)
        return dq, dk, dv


def triton_causal_attention(q, k, v):
    return TritonCausalAttention.apply(q, k, v)
```

### 10.5 Padding 与数值边界的解释

forward 对最后一个 query block 中超出 N 的行加载 0，并不保存这些行。因为所有行在第一块都能看到 key 0，所以 online softmax 初始化时有有限最大值，不出现全 `-inf` 的减法。这个论证依赖 `N≥1` 和当前 causal 规则；加入 padding、任意 mask 或空序列后要重新设计。

backward 的 `allowed` 同时检查 query、key 和 causal，超界位置概率设为 0。所有合法内存访问必须有 mask；不能只在最后 store 时 mask，却在 load 时读越界。

低精度 `p.to(...)` 会影响输出与梯度误差。测试容差应基于 dtype、输入尺度和参考误差分布决定；不能把很宽的容差当作“训练可用”的证明。

### 10.6 验证矩阵

至少覆盖 `T={1,31,33,63,65,127,257,1024}`、`D={64,128}`、不同 B/H、FP16／BF16。每次使用随机 dO 比较 `O,dQ,dK,dV`，报告 max absolute error 与 normalized L2 error。

参考可以使用 FP32 手写 Attention 与 PyTorch Autograd，也加入 SDPA。FP16／BF16 kernel 不支持直接 FP64 gradcheck；可以对 FP64 数学实现做 gradcheck，再验证 kernel 对它的近似误差。

测试阶段进一步检查：重复运行无状态污染、最后一块不越界、因果性、极端但有限输入、non-contiguous 输入经 wrapper 后一致、训练一步 loss 与参考接近。

把 kernel 接入小模型跑几十步，比较相同初始化、数据、学习率下的 loss 曲线与梯度范数；正确输出与正确训练轨迹是两道检查。

### 10.7 优化顺序

1. 跳过纯未来的 causal blocks，减少无用 matmul。
2. 扫描 BM／BN、num_warps、num_stages，保存 shape 对应配置。
3. 检查寄存器 spill、occupancy、访存合并、shared memory。
4. 减少非 matmul 的缩放／同步，参考 FA2 的 work partition。[R17](12_模板公式与参考.md#r17)
5. 将 delta precompute 也融合／写为 Triton，避免多次临时转换。
6. 支持任意 strides、GQA、padding、variable-length 时逐个增加并重验。
7. 分别测试 forward、backward、整块 Transformer 和端到端训练。

调参不能使用测试集性能作为唯一选择依据；另留 shapes 检查泛化。小序列上自写 kernel 可能更慢，真实结论应写“在哪些 shapes 有收益”。

<a id="chapter-11"></a>
## 11. 多卡训练与多卡推理

### 11.1 并行方式

| 方式 | 怎么分 | 内存收益 | 通信／调度代价 |
| --- | --- | --- | --- |
| Data parallel／DDP | 每卡完整模型，不同 batch | 模型状态不省 | 同步梯度 |
| ZeRO-1 | 分 optimizer states | 状态降为约 1/W | 梯度归约、参数同步 |
| ZeRO-2 | 再分梯度 | 梯度／状态下降 | reduce-scatter／参数同步 |
| ZeRO-3／FSDP | 再分参数 | 模型状态进一步下降 | 参数 all-gather、梯度 reduce-scatter |
| Tensor parallel | 分同一层矩阵／heads | 每层权重下降 | 层内频繁 collective |
| Pipeline parallel | 分层到不同设备 | 分层保存 | pipeline bubbles 与 micro-batch 调度 |
| Sequence／context parallel | 分序列相关计算 | 激活／长序列压力下降 | 额外通信与 attention 设计 |

对约 0.1B 模型，DDP 是多卡训练首选实验基线；FSDP 主要用于学分片机制，TP 主要用于测清楚小模型的通信代价。模型能轻松装入单卡时，多卡不保证更省钱。

### 11.2 DDP 的数学

各 rank 计算自己的梯度，all-reduce 后默认对 world size 平均。若每 rank 的有效样本数相等，各 rank 的平均 loss 可直接 backward。

若有效 token 数不同，设全局有效 token 数为 M，每 rank 本地 loss sum 为 `S_r`，应该 backward：

$$
\mathcal L_r=W S_r/M.
$$

DDP 平均后是 `Σ_r∇S_r/M`，恰好对应全局 token mean。忽略 W 会让梯度缩小 W 倍；各 rank 各自取 mean 会错误地等权不同大小的数据。

### 11.3 梯度累积与 no_sync

前 A−1 个 micro-step 用 `no_sync()`，最后一步同步。**forward 和 backward 都要在 no_sync context 中**。[R20](12_模板公式与参考.md#r20) 同步之后再 clip、optimizer step；所有 rank 的 collective 顺序必须一致。

以下参考函数用于已初始化的 DDP 和 PyTorch optimizer，batch 已搬到 GPU。它突出全局归一化，不是完整多进程入口：

```python
import contextlib
import torch.distributed as dist


def ddp_train_update(ddp, microbatches, optimizer, clip=1.0):
    # Every rank must have the same number of microbatches.
    device = microbatches[0][0].device
    local_count = sum(int(mask.sum()) for _, _, mask in microbatches)
    global_count = torch.tensor(local_count, dtype=torch.long, device=device)
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if global_count.item() == 0:
        raise ValueError("empty global update")
    factor = dist.get_world_size() / global_count
    optimizer.zero_grad(set_to_none=True)
    for i, (ids, labels, mask) in enumerate(microbatches):
        sync = i == len(microbatches) - 1
        ctx = contextlib.nullcontext() if sync else ddp.no_sync()
        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = ddp(ids)
            z = logits.float()
            logp = z.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            logp = logp - torch.logsumexp(z, -1)
            loss_sum = -(logp * mask).sum()
            (loss_sum * factor).backward()
    torch.nn.utils.clip_grad_norm_(ddp.parameters(), clip)
    optimizer.step()
```

mask 必须已排除 invalid labels。生产版从 CPU batch metadata 预先统计有效数，减少 `.item()` 同步。AMP／非有限梯度与跳步也要全 rank 协调，不能只有一张卡自行跳过 optimizer。

典型进程入口设置 `LOCAL_RANK` 对应设备，初始化 NCCL，构造 DDP，使用各 rank 的数据 shard，最后销毁进程组。命令示意：

```bash
torchrun --standalone --nproc_per_node 2 scripts/train_ddp.py --config configs/pretrain.yaml
```

`scripts/train_ddp.py` 是项目中待实现的入口，不是本文已交付的脚本。

### 11.4 自己测 collective 与重叠

先独立测不同 tensor 大小、dtype、GPU 数、跨节点的 all-reduce／all-gather／reduce-scatter。每轮 barrier、预热、同步，保存带宽与延迟。

ring all-reduce 每 rank 传输约：

$$
2(W-1)S/W\ \text{bytes},
$$

`S` 是梯度 tensor bytes。小 tensor 受 launch／latency 影响，大 tensor 更受链路带宽限制。`S/time` 的算法带宽和乘上通信系数的 bus bandwidth 是不同口径。

逐步实现以下 DDP 教学对照：

1. backward 完毕后逐 Parameter all-reduce。
2. 扁平化梯度一次 all-reduce。
3. autograd hook 中异步归约。
4. 按 bucket 归约并与 backward 重叠。
5. 与 PyTorch DDP 对比 bucket 大小和等待时间。

异步调用返回不等于结果可用；optimizer 前必须等待相关 work。不要同时写入还在通信的 buffer。tied 参数注册／同步一次，所有 rank 参数顺序一致。

### 11.5 分片显存与 FSDP

以 FP32 状态简化估算：DDP 约 `16N`；ZeRO-1 约 `8N+8N/W`；ZeRO-2 约 `4N+12N/W`；全分片约 `16N/W` 加 all-gather、激活和临时空间。低精度／master weights 策略不同要重算。

FSDP2 用 `fully_shard` 给模块建立参数 all-gather 和梯度 reduce-scatter 的生命周期。通常先对各 block 调用，再对 root 调用；optimizer 在分片之后构造。[R21](12_模板公式与参考.md#r21)

接口位置随 PyTorch 版本变化，下面是当前公开文档形式的示意，安装版本须核对：

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

# Requires an initialized process group and a compatible PyTorch version.
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.blocks:
    fully_shard(block, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

全分片并非只把 DDP 的一个 collective 换名：reshard 后 backward 可能需要再次 all-gather，通信量与缓冲生命周期不同。模型很小时，细粒度 all-gather 的延迟会抵消省显存的好处。对 wrap 粒度、prefetch、reshard_after_forward、reduce dtype 与 checkpoint 保存分别测量。

分片梯度的裁剪要使用全局 norm：概念上各 rank 对本地分片求平方和，归约后开方，再以相同系数裁剪；复制的参数不能重复计入。优先使用锁定版本支持 DTensor／分片的工具，不能直接对每卡局部 norm 独立 clip，否则更新不等价于原模型。

### 11.6 强扩展与弱扩展

定义单卡吞吐 R1，W 卡吞吐 RW：

$$
S_W=R_W/R_1,\quad E_W=S_W/W.
$$

**强扩展**固定总工作量和全局 batch，增加卡数，降低每卡 batch／累积；主要回答“同一训练多久完成”。**弱扩展**固定每卡工作量，全局 batch 随 W 增长；主要回答“资源增加能承载多少吞吐”，其优化行为也可能变化。

| W | global tokens/update | local micro-batch | accum | tokens/s | step ms | efficiency | cost/token | val loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65,536 | 实测配置 | 实测配置 | 待测 | 待测 | 1.00 | 待算 | 待测 |
| 2 | 65,536 | 调整 | 调整 | 待测 | 待测 | 待算 | 待算 | 待测 |
| 4 | 65,536 | 调整 | 调整 | 待测 | 待测 | 待算 | 待算 | 待测 |

还要记录 PCIe／NVLink／跨节点网络、GPU clocks、CPU／磁盘是否共享。跨卡数据重复、dataloader bottleneck 和小 micro-batch 是常见原因。

### 11.7 多卡推理

**副本并行**：每卡完整模型，不同请求，各自 KV cache，路由器分配。**Tensor parallel**：同一个请求跨卡分层矩阵，decode 每层有通信。小模型通常值得先测副本并行，再测 TP；结果依赖 GPU／互联与负载。

默认 12 heads 在常见切分下可测 TP=2、4；TP=8 可能不满足 heads／KV heads 的整除限制，不能无条件使用。GQA 和 engine 的分配策略可能有额外约束。

多卡推理实验固定模型、dtype、prompt／output 分布、请求总量、采样、总并发与到达率。另做容量实验扫描总并发，比较每种部署在相同 SLO 下的 goodput。

| 部署 | 测什么 | 同时报告 |
| --- | --- | --- |
| 1 卡单副本 | 基准 | TTFT、TPOT、P95、吞吐、费用 |
| 2／4 卡副本 | 多请求吞吐 | 路由开销、每副本负载、总体 goodput |
| 2／4 卡 TP | 单请求与系统表现 | collective 时间、cache／权重内存、SLO |
| 不同 batch／cache 策略 | 调度收益 | 质量、排队、OOM／重试 |

除了 speedup，计算每千输出 tokens 费用和每个成功任务费用。4 卡吞吐 3 倍可以缩短排队，但在相同卡时单价下单位成本可能变差；报告应让使用者看到这种权衡。
