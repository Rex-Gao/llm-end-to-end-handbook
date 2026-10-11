# 08 · RLVR与对照

[阅读说明](../README.md) · [完整手册](../大模型全链路知识手册.md) · [上一册：SFT与DPO](07_SFT与DPO.md) · [下一册：环境与Harness](09_环境与Harness.md)

<a id="chapter-16"></a>
## 16. RLVR：从可验证结果学习

### 16.1 奖励来自哪里

RLVR 用可执行验证给奖励：数学表达式是否正确、代码是否通过可信测试、结构是否满足约束、环境是否达到目标。RLHF 常使用人类偏好或学习出来的 reward model。二者都可以用 policy gradient，但信号可靠性与错误模式不同。

“能写 reward 函数”不等于“奖励完整表达任务”。单元测试可能漏掉行为，parser 可能拒绝等价答案，最终文件可能正确而副作用错误。先审计 verifier，再训练策略。

### 16.2 不使用 eval 的表达式 verifier

下面用 Python AST 与 Fraction，只允许整数常量、加减乘除、括号，要求给定数字各用一次；限制长度、节点、深度与运算规模。适合小型 Countdown-like 任务，不执行任意 Python。

```python
import ast
from collections import Counter
from fractions import Fraction


def verify_expression(expression, numbers, target):
    if not isinstance(expression, str) or len(expression) > 512:
        return False
    if not numbers or len(numbers) > 8:
        return False
    if any(type(n) is not int or not 0 <= n <= 1_000_000 for n in numbers):
        return False
    available = Counter(numbers)
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            return False

        def visit(node, depth=0):
            if depth > 20:
                raise ValueError("too deep")
            if isinstance(node, ast.Constant) and type(node.value) is int:
                n = node.value
                if available[n] <= 0:
                    raise ValueError("number unavailable")
                available[n] -= 1
                value = Fraction(n)
            elif isinstance(node, ast.BinOp):
                left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                elif isinstance(node.op, ast.Div) and right != 0:
                    value = left / right
                else:
                    raise ValueError("operator forbidden or division by zero")
            else:
                raise ValueError("node forbidden")
            if value.numerator.bit_length() > 128 or value.denominator.bit_length() > 64:
                raise ValueError("arithmetic limit")
            return value

        value = visit(tree.body)
        return value == Fraction(target) and all(v == 0 for v in available.values())
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError, RecursionError):
        return False
```

表达式 `2+3*7` 对数字 `[2,3,7]` 和目标 23 应通过；直接写 23、额外数字、重复使用、除零、属性访问、函数调用都应失败。使用 Counter 是为了正确处理重复数字。

它不接受所有数学等价语法，因此任务指令必须明确语法范围。parser 错误应记 format failure，不能一律当作数学能力失败。目标值来自受信 task，建议为整数／Fraction。

### 16.3 生成任务与课程难度

从数字构造合法表达式，计算 target，再隐藏表达式。独立搜索器确认可解；清除重复题与零乘法捷径；按数字 multiset、target、生成结构切分，避免只换顺序就进入测试。

难度可按数字数、运算深度、除法比例、解的数量与初始成功率分层。先选有非零成功概率的任务。全组失败时，组相对 RL 没有有效排序信息；用容易课程或明确记录成本的短 SFT warm-start 帮助启动。

### 16.4 REINFORCE 与组优势

$$
\nabla_\theta\mathbb E[R]=\mathbb E[(R-b)\nabla_\theta\log\pi_\theta(y\mid x)].
$$

序列 log-prob 是各生成 token 的和，reward／advantage 要 detach。baseline 应符合使用条件，不能随意把依赖当前动作的量放进去后声称无偏。

经典 GRPO 对同一 prompt 的 G 个回答计算：

$$
A_i=(r_i-\bar r)/(\operatorname{std}(r)+\epsilon).
$$

```python
def group_advantages(rewards, normalize_std=True, eps=1e-6):
    # [n_prompts,G], G >= 2
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("need at least two samples per group")
    r = rewards.float()
    if not torch.isfinite(r).all():
        raise ValueError("nonfinite reward")
    centered = r - r.mean(-1, keepdim=True)
    if normalize_std:
        centered = centered / (r.std(-1, keepdim=True, unbiased=False) + eps)
    return centered.detach()
```

全对／全错组优势为 0；ε 不会创造学习信号。G=1 的自身均值基线也为 0，需要其他 baseline。population／sample std 口径必须固定。[R28](12_模板公式与参考.md#r28)

去 std 只做中心化，可以避免问题级方差改变难度权重。[R29](12_模板公式与参考.md#r29) leave-one-out baseline 在不做 std 时产生组中心化优势的 `G/(G-1)` 倍；不是所有变体都具有完全相同的估计性质。

### 16.5 三个策略与真实行为分布

`π_old` 生成本轮 rollout；`πθ` 正在更新；`π_ref` 是可选固定参考，用于 KL。三者不能混用。

$$
\rho_{i,t}=\exp(\log\pi_\theta(y_{i,t}\mid c_{i,t})-
\log\pi_{old}(y_{i,t}\mid c_{i,t})).
$$

上下文 `c` 必须与生成时相同。入门固定 temperature=1、top-p=1；temperature／top-p 修改分布后，原始 logits 的概率不直接等于真实行为概率。

首次更新前 ratio 数值为 1，但它对 θ 的导数不为 0，仍可学习。old log-probs、advantage 和 rewards 不求梯度。模型更新后再复用 rollout，clip 才可能触发。

### 16.6 Clip 与 loss 归约

$$
\ell=-\min(\rho A,\operatorname{clip}(\rho,1-\epsilon_{low},1+\epsilon_{high})A).
$$

```python
def clipped_pg_loss(new_logp, old_logp, advantages, mask,
                    eps_low=0.2, eps_high=0.2,
                    reduction="sequence", fixed_length=None):
    if new_logp.shape != old_logp.shape or mask.shape != new_logp.shape:
        raise ValueError("token shapes mismatch")
    m = mask.to(new_logp.dtype)
    lengths = m.sum(-1)
    if torch.any(lengths == 0):
        raise ValueError("empty response")
    delta = torch.where(mask.bool(), new_logp - old_logp.detach(), 0.0)
    ratio = delta.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError("nonfinite ratio")
    adv = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * adv,
                               ratio.clamp(1-eps_low, 1+eps_high) * adv)
    token_loss = -surrogate * m
    if reduction == "sequence":
        return (token_loss.sum(-1) / lengths).mean()
    if reduction == "token":
        return token_loss.sum() / lengths.sum()
    if reduction == "fixed" and fixed_length is not None and fixed_length > 0:
        return token_loss.sum() / (len(lengths) * fixed_length)
    raise ValueError("invalid reduction")
```

`sequence`：每回答先平均再按回答平均，短答每 token 权重更大。`token`：全 batch 按有效 token 平均，长答总权重更大。`fixed`：按固定预算常数归一化，减少单条长度造成的缩放变化，梯度尺度也改变。比较时重新调 LR、记录真实公式。[R29](12_模板公式与参考.md#r29)、[R30](12_模板公式与参考.md#r30)

### 16.7 KL、熵与过长处理

一种 sampled-token KL 估计使用 `u=logπ_ref-logπθ`，计算 `exp(u)-u-1`。在正确当前策略采样的条件下，其期望对应 KL；它不是任意离策略样本上的精确全词表 KL。加入 clip／KL／长度处罚后目标均发生变化。

估计出 KL 数值与得到该期望目标的正确梯度并非同一件事：采样分布依赖策略，直接将采样点视为常量求导会涉及额外近似。记录 KL 的方向、采样分布、importance weights 与 detach 位置，按已验证的优化配方实现，不能只看函数值非负就认定训练正确。

监测 entropy、KL、clip fraction、ratio 分位数、梯度范数、reward mean／std、有效组比例、正确／错误输出长度和截断率。只画 reward 不够。

max-token 截断应有独立标签。比较截断 loss mask、明确失败、soft overlong penalty，避免对被截掉的未完成推理产生错误信号。被过滤的 rollout 仍消耗计算，必须计费。

### 16.8 算法构件与消融

| 配方 | 关键构件 | 检查点 |
| --- | --- | --- |
| GRPO | 组 std、序列内平均、clip、可选 KL | 长度／难度权重 |
| Dr.GRPO | 去组 std、固定常数归约 | 长度偏置与梯度尺度 |
| DAPO | 非对称 clip、动态采样、token-level loss、过长处理 | 熵、过滤比例、总成本 |
| PPO | critic、GAE、clip | critic 误差与显存 |
| Expert Iteration | 采样正确解后 SFT、迭代 | 自蒸馏、探索覆盖 |

DAPO 论文数值对应其大模型／任务，不能直接投射到本模型。[R30](12_模板公式与参考.md#r30) RL 是否拓展能力是需要独立评测的问题，不能无条件断言“只选择已有解”或“必然创造新能力”。

### 16.9 训练循环合同

```text
冻结本轮行为策略版本
  -> 从 TRAIN 采样 prompts
  -> 每题 G 个 responses，保存原 token IDs 与 logprobs
  -> verifier 打分并保存分项和终止原因
  -> 按正确 group IDs 计算 advantage
  -> 用生成时上下文 teacher forcing，更新 policy
  -> 检查 ratio 漂移，限制每批复用次数
  -> 同步新权重到 rollout worker
  -> 固定 DEV 评测，最后独立 TEST
```

不能把生成 IDs 解码再重新 encode 当原轨迹；同一字符串可能由不同 token 序列生成，重编码会改变行动概率。保留 stop token，EOS 是否参与 loss 要一致。

vLLM 与 trainer 的 dtype／kernel 可能不同，先检查同权重下的 log-prob 差异。每条轨迹保存 policy version，防止异步混入旧权重。

教学起点可为 16 prompts × G=4、max response 128、2～3 个数字、每轮 1 次更新；再扫 LR、G、std、归约、KL、clip、复用与课程。任何数字都需 pilot 验证。

模型生成的格式、推理与工具动作属于 policy tokens；环境返回的观测不是 policy 行动，loss mask 为 0。reward 涨而 oracle 不涨时，检查 parser、截断、重复题和格式奖励，先修奖励再增加预算。

<a id="chapter-17"></a>
## 17. 同一基座的 SFT／DPO／RLVR 对照

### 17.1 两类问题

第一组从同一预训练 C0 直接分叉，研究示范、偏好和在线结果奖励各带来什么；direct RL 可能暴露格式和探索困难。第二组先构造公共协议 warm-start C1，再分别继续 SFT／DPO／RLVR，减少“只是先学会格式”的混淆。

两组回答不同问题，图表分别报告。所有主分支固定 Tokenizer、架构、精度、全参数训练；LoRA 和更大外部基座单列。

| 分支 | 起点 | 新监督 | 目的 |
| --- | --- | --- | --- |
| C0 | 自训预训练模型 | 无 | 任务／语言基线 |
| C0-SFT | C0 | 正确示范 | 直接监督 |
| C0-DPO | C0，ref=C0 | 偏好对 | 直接偏好训练 |
| C0-RLVR | C0 | verifier rewards | 直接 RL 启动 |
| C1 | 指定 C0-SFT checkpoint | 公共 warm-start | 统一交互能力 |
| C1-SFT-more | C1 | 更多示范 | 增量监督对照 |
| C1-DPO | C1，ref=C1 | 偏好对 | 偏好增量 |
| C1-RLVR | C1 | 在线验证奖励 | RL 增量 |
| C1-DPO-RLVR | C1-DPO | RLVR | 组合 recipe，单列 |

### 17.2 共享任务与监督信息

建立共享 TRAIN prompt pool：整数表达式、短约束、结构化输出与简单工具流程。SFT 用已验证示范，DPO 用同 prompt 的优劣回答，RLVR 在同 pool 采样。

同 pool 不等于监督等价：教师完整解法、偏好标签与结果奖励提供的信息不同。记录教师／judge／oracle 的版本和调用数，不能把教师蒸馏的收益归给没有教师的 RL。

### 17.3 公平预算

分别提供“固定训练 processed tokens／FLOPs”和“固定总 GPU hours／费用”视角。后者计入 reference、双回答、rollout、判官、工具、沙箱与失败重试。动态采样丢掉的 group 也计成本。

只有 optimizer steps 相同不公平；所有成本完全相同又常不可行，因此明确资源轴并展示 learning curve。

### 17.4 评测与解释

必须测同分布／OOD success、format rate、pass@1／@4／更大 k、长度、entropy、基础 NLL 保持、工具合法率、false_pass、cost/success 与训练稳定性。OOD 改变数字范围、组合、任务长度与工具错误，不只换措辞。

结论限定“本基座、题族和预算下”的差值与区间。direct RL 不能启动应如实报告，再分析基座支持、奖励密度和探索；warm-start 后成功不能当作 C0 的原始能力。

Tülu 3 的 SFT→DPO→RLVR recipe 是设计参照，不替代这里的独立对照。[R27](12_模板公式与参考.md#r27) 保存全部失败轨迹，按“格式、知识、搜索、工具、记忆、验证”分类，再决定下一轮工作。
