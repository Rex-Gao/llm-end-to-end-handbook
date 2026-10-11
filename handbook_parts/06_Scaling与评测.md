# 06 · Scaling与评测

[阅读说明](../README.md) · [完整手册](../大模型全链路知识手册.md) · [上一册：Triton与多卡](05_Triton与多卡.md) · [下一册：SFT与DPO](07_SFT与DPO.md)

<a id="chapter-12"></a>
## 12. 拟合 Scaling Law 并外推

### 12.1 先明确要预测什么

本章预测固定数据分布、Tokenizer、模型家族、优化配方下的 heldout language-model loss，以及固定训练计算预算下的 N／D 配置。它不能直接保证数学、代码或 Agent 成功率，也不能用预训练 loss 的幂律解释所有后训练变化。

`N` 用 total、non-embedding 还是 activated 参数，`D` 是总消费还是独特 token，`C` 是否含 attention／重算，都必须固定。小模型的 Embedding 占比尤其大，直接套大模型参数近似容易出偏差。

### 12.2 Chinchilla 型损失模型

$$
\mathcal L(N,D)=E+A N^{-\alpha}+B D^{-\beta},\quad A,B,\alpha,\beta>0.
$$

E 是拟合出来的渐近项，不等于已经知道了真实不可约语言熵。该模型把容量不足和数据／训练不足近似分开。它是经验模型，需在自己的范围中验证。[R24](12_模板公式与参考.md#r24)

原论文与后续复现讨论也表明，拟合的损失缩放、优化停止条件、舍入与置信区间会明显改变外推；不能只抄一组系数。[R25](12_模板公式与参考.md#r25)

### 12.3 固定 compute 的最优点

若 `C=κND`，常取近似 `κ=6`，则 `D=C/(κN)`。最优点满足：

$$
\alpha A N^{-\alpha}=\beta B D^{-\beta}.
$$

因此：

$$
N_{opt}=\left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}
\left(\frac C\kappa\right)^{\beta/(\alpha+\beta)},
$$

$$
D_{opt}=\left(\frac{\beta B}{\alpha A}\right)^{1/(\alpha+\beta)}
\left(\frac C\kappa\right)^{\alpha/(\alpha+\beta)}.
$$

α≈β 时两者约随 `C^0.5` 增长。常见的“20 tokens／parameter”是某类 compute-optimal 结果的量级，不是所有模型、语料、重复数据或服务目标的自然常数。考虑长期推理成本时，训练更小模型更多 tokens 可能合算。

### 12.4 实验家族与预算分配

教学候选如下，参数量取近似值，实际必须由代码统计：

| dim | layers | heads | FFN | 总参数量级 | 用途 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 4 | 2 | 384 | 约 3M | 快速扫描／验证实现 |
| 256 | 6 | 4 | 704 | 约 9M | 小规模曲线 |
| 384 | 8 | 6 | 1024 | 约 20M | 中等规模 |
| 512 | 12 | 8 | 1408 | 约 47M | 留出规模／外推校验 |
| 768 | 12 | 12 | 2048 | 97.5M | 主线目标模型 |

保持词表与主要架构一致，每个模型跑至少 3 个 D 或 3 个 compute budgets。只跑各模型一次 `D=20N`，N 和 D 强耦合，很难可靠辨识 α 与 β。

预算有限可先用 3M／9M／20M 与 5M／15M／45M tokens 的网格探索，再把 47M 的运行作为留出。它只能支持有限范围的判断；用它直接外推 97.5M／2B 时跨度很大，区间应宽，必须增加目标附近 pilot。

预算较充足再增加数据轴与 compute 轴的覆盖，把总计划 FLOPs 求和后再启动。最佳学习率应先通过有限搜索确定；欠调的某个模型会让拟合误判为“规模越大反而越差”。

### 12.5 IsoFLOP 方法

给定几个预算 C，每个预算选择多个 N，按 `D=C/(κN)` 训练。对每条 loss-vs-logN 曲线找到内部最小点，再拟合：

$$
\log N_{opt}=a\log C+b.
$$

如果最优点出现在最小／最大 N 边界，说明搜索范围不足；不能把边界点当真最优。可以用最小点附近的二次曲线拟合辅助定位，但要检查曲率、数据噪声和落点是否在观测范围内。

离散 N、最短有效训练长度、LR schedule 和优化器 warmup 会破坏理想等 FLOPs。记录实际 C，不要只记录计划值。

### 12.6 多起点稳健拟合代码

以下使用 NumPy／SciPy，输入必须是独立训练运行的真实结果。`p=[log E,log A,log B,α,β]`，在 log-loss 空间拟合，用 Huber 降低异常点影响。`f_scale=0.01` 是教学选择，要做敏感性分析。

```python
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp


def scaling_log_loss(p, n, d):
    e, a, b, alpha, beta = p
    terms = np.stack((np.full_like(n, e),
                      a - alpha * np.log(n),
                      b - beta * np.log(d)), axis=0)
    return logsumexp(terms, axis=0)


def fit_scaling(n, d, loss):
    n, d, loss = (np.asarray(x, dtype=float) for x in (n, d, loss))
    if not (n.shape == d.shape == loss.shape) or n.ndim != 1:
        raise ValueError("expected equal one-dimensional arrays")
    if len(n) < 8 or np.any(n <= 0) or np.any(d <= 0) or np.any(loss <= 0):
        raise ValueError("need enough positive observations")
    observed = np.log(loss)
    upper_e = np.log(loss.min() * 0.999)
    bounds = ([-20, -100, -100, 0.01, 0.01], [upper_e, 100, 100, 2, 2])
    best = None
    for alpha in (0.2, 0.35, 0.6, 1.0):
        for beta in (0.2, 0.35, 0.6, 1.0):
            E0 = loss.min() * 0.25
            contribution = max((np.median(loss) - E0) / 2, 1e-6)
            initial = [np.log(E0),
                       np.log(contribution) + alpha * np.log(np.median(n)),
                       np.log(contribution) + beta * np.log(np.median(d)),
                       alpha, beta]
            result = least_squares(
                lambda p: scaling_log_loss(p, n, d) - observed,
                initial, bounds=bounds, loss="huber", f_scale=0.01,
                max_nfev=5000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if result.success and (best is None or result.cost < best.cost):
                best = result
    if best is None:
        raise RuntimeError("all fits failed")
    return best.x, best


def scaling_optimum(p, compute, kappa=6.0):
    if compute <= 0 or kappa <= 0:
        raise ValueError("positive compute required")
    _, logA, logB, alpha, beta = p
    logN = ((np.log(alpha / beta) + logA - logB)
            + beta * np.log(compute / kappa)) / (alpha + beta)
    N = np.exp(logN)
    return N, compute / (kappa * N)
```

不应把“优化器收敛”当成“缩放律已成立”。数据点过少或相关，模型也能拟合出漂亮曲线。检查多个起点是否给出相近预测，参数是否卡 bounds，E 是否合理，外推是否对少数点过度敏感。

### 12.7 不确定性与外推验证

至少做：

1. 留出最大预算或最大模型，先预测再运行／揭盲。
2. 画 residual-vs-N、residual-vs-D、residual-vs-C。
3. 对独立 run 做 bootstrap，重复拟合，给 loss／Nopt／Dopt 的分位区间。
4. 多个 checkpoints 来自同一 run 时，按 run 分组重采样，避免伪造样本量。
5. 比较 Huber scale、删除异常点、数据混合变化的敏感性。
6. 加入邻近目标的 pilot，缩小外推跨度。

报告参数区间还不够，应直接报告目标预算的预测区间。给定离散硬件友好形状，把连续 Nopt 转成多个邻近候选，按实际 FLOPs 与吞吐再选择。

### 12.8 实用外推报告

写清：拟合范围、目标跨度、固定项、N/D/C 口径、训练 schedule、运行数量／seeds、holdout error、bootstrap、失败点与局限。不能写“Scaling Law 证明本模型会达到某个 Agent 成功率”。

若目标由“最低预训练 loss”变成“最低服务成本且准确率达到阈值”，需要建立训练成本 + 推理需求的联合目标；这可能推荐更小、训练更久的模型。

<a id="chapter-13"></a>
## 13. 评测与可信实验设计

### 13.1 三层评测

| 层次 | 回答的问题 | 指标 |
| --- | --- | --- |
| 语言建模 | 预测正文是否更好 | NLL、PPL、BPB，按语言／来源分项 |
| 单轮任务 | 答案／格式／约束是否正确 | exact match、verifier success、format rate |
| Agent episode | 多步行动能否达成目标 | success、步骤、成本、恢复、约束违例、超时 |

语言建模要按有效 tokens 汇总 NLL。`PPL=exp(mean NLL)` 仅在相同 Tokenizer 和评分协议下比较。跨 Tokenizer 优先同原文 BPB。不同任务表现不能只合成一个没有权重解释的总分。

### 13.2 数据与评测协议冻结

训练集优化参数，开发集选择超参，测试集用于最终比较。测试集不能用于挑选 judge prompt、reward weights、解析器特殊规则或失败恢复策略。所有这些都是会影响成绩的可训练／可选择部件。

每个评测版本保存 task IDs、输入、参考／hidden tests、harness config、tool schema、verifier hash、采样参数、预算、模型／Tokenizer hash。换 prompt 或 harness 等于换评测条件。

### 13.3 Pass@k 与重复可靠性

每题采样 n 个回答，其中 c 个成功，`n≥k` 时常用无放回估计：

$$
\widehat{pass@k}=1-\frac{\binom{n-c}{k}}{\binom nk}.
$$

它衡量多次尝试中至少一次成功的机会；必须报告采样成本和验收器。k 次采样使用外部强 judge 选择，也是一种额外系统能力。

`pass^k` 常表示 k 次独立试验全部成功的可靠性；应优先实测重复任务的联合成功。若试验相关，不能简单把 `pass@1` 的均值做 k 次方当真值，也不能混称为 pass@k。

RL 可能让 pass@1 上升、候选多样性下降，较大 k 的探索能力未必上升。比较 base 与 posttrained 时同时测不同 k、entropy 和正确策略种类；“更会选已有解”与“能力覆盖扩大”需要不同证据。

### 13.4 样本量与统计

任务成功率是随机量。几十题、单次 seed 的几个百分点变化可能只是噪声。二项成功率可用 Wilson 区间；跨模型同题比较用 paired bootstrap 或成对检验，不要把同一题的多个 rollout 当成多个独立题目。

```text
paired bootstrap:
    sample task IDs with replacement
    preserve both models' outcomes for each sampled task
    compute score_A - score_B
    repeat, then report interval and full distribution
```

将环境初始化、模型采样与训练 seed 区分。训练 seed 的变化体现训练稳定性；评测 seed 的变化体现采样和环境随机性。预算有限，先在少数配置做探索，再对最终候选增加重复。

### 13.5 长度、格式和成本是必要分项

报告正确／错误回答各自的平均与 P95 长度，格式合法率、截断率、拒答率、空回答、工具错误、超时、假通过。奖励增长但长度暴涨，可能是策略利用了归一化或 judge 的长度偏好。

真实成本包括训练、rollout、参考模型、判官、工具、沙箱 CPU、存储、网络与重试。记录“每个成功任务 tokens／GPU 秒／总费用”，防止多采样、长输出造成看起来更高的分数。

### 13.6 Verifier 与 oracle 分开

训练 verifier 可以是便宜且不完美的检查器；独立 oracle 应更严格，例如 hidden tests、符号精确验证或人工复核。oracle 不进入 actor 可见上下文。

定义：

$$
false\_pass=P(oracle=0\mid verifier=1).
$$

另报告整体假阳性 `P(verifier=1,oracle=0)`、假阴性和 verifier coverage。没有正例时 conditional false_pass 不定义，不应填 0 假装安全。

### 13.7 受控实验与消融

固定初始 checkpoint，只改变一个关键因素；给每个分支记录训练数据／tokens／FLOPs／GPU hours／rollout tokens。公平性可以有多个视角：相同优化 tokens、相同 GPU hours、相同总费用、相同任务监督来源。不能假定它们能同时完全相等。

验证 harness 改动时固定模型；验证模型改动时固定 harness；研究联动时提供完整因子实验。每次最终测试保存全部失败 trace，而不只保存成功样例。
