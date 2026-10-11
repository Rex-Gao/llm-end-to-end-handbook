# 07 · SFT与DPO

[阅读说明](../README.md) · [完整手册](../大模型全链路知识手册.md) · [上一册：Scaling与评测](06_Scaling与评测.md) · [下一册：RLVR与对照](08_RLVR与对照.md)

<a id="chapter-14"></a>
## 14. SFT：学习示范与交互协议

### 14.1 SFT 的作用

SFT 在示范回答上做最大似然训练，适合让模型学习回答风格、格式、工具协议、基本任务解法和失败恢复。它提供了模型原先未必会采样到的示范；代价是数据质量、覆盖和分布偏差会直接进入策略。

对约 0.1B 模型，先教会短答案、结构化输出和简单工具动作，再引入长推理。大模型的长 CoT 不一定是小模型容易吸收的最佳教学形式；可以比较短证明、分解动作和精简推理。

### 14.2 数据格式与 chat template

建议保存原始 messages 和标准化训练 token 序列，二者都有版本：

```json
{
  "task_id": "arith_train_000001",
  "messages": [
    {"role": "user", "content": "计算 17 + 25，只输出整数。"},
    {"role": "assistant", "content": "42"}
  ],
  "source": "受控生成器",
  "verifier": "integer-exact-v1",
  "verified": true,
  "split": "train"
}
```

模板示意：`<|bos|><|user|>问题<|end_turn|><|assistant|>回答<|end_turn|>`。控制 token 应由 renderer 显式插入 IDs，用户正文用 ordinary encoding，避免字面字符串变成角色边界。

推理时同样使用这个 renderer，并以 assistant 角色起始位置结束 prompt。模板改变会影响模型条件分布，必须纳入 checkpoint 和 eval 版本。

### 14.3 Response-only loss 的正确移位

把完整序列 token IDs 记为 `s`，同长度 `response_mask` 中 assistant 输出／终止 token 为 1，其他为 0。输入 `s[:-1]`，目标 `s[1:]`，所以 loss mask 必须用 `response_mask[1:]`。

```python
def response_log_probs(model, input_ids, response_mask):
    # Right-padded sequences; at least one prompt token precedes the response.
    logits = model(input_ids[:, :-1]).float()
    targets = input_ids[:, 1:]
    logp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logp = logp - torch.logsumexp(logits, dim=-1)
    mask = response_mask[:, 1:].to(logp.dtype)
    return logp, mask


def sft_loss(model, input_ids, response_mask):
    logp, mask = response_log_probs(model, input_ids, response_mask)
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("no assistant tokens")
    return -(logp * mask).sum() / count
```

prompt 没有直接目标 loss，不表示它的内部激活没有梯度：生成 response 的预测仍然通过上下文反向传播。多轮轨迹可以监督多个 assistant turn；tool outputs、system 和用户消息通常不作为 assistant 目标。

本函数只处理右侧 padding；有效 query 不看后面的 pad，所以参考模型可用常规 causal mask。left-padding、packed conversations、任意 position IDs 需要扩展模型 attention mask。不同样本 pack 进一条序列时应明确是否允许相互关注。

### 14.4 Tokenize 边界不能凭字符长度推断

如果分别 tokenize prompt 和回答再拼接，编码可能不同于完整文本的 BPE。使用明确 special boundary，或者由 renderer 在拼接过程中返回每个 token 的角色区间。不能把 prompt 字符数当 prompt token 数。

截断后要重新检查至少一个监督 token、EOS 是否保留、tool call 是否完整。空 response 或被截光的 response 应拒绝或单独处理，不能出现除零。

### 14.5 构造适合小模型的数据

推荐混合：短算术、整数表达式、结构化信息抽取、JSON schema 遵循、工具选择、工具结果解释、出错重试、最终提交。每类都有可验证标准。

数据覆盖应包括：正常路径、无解／不满足条件、重复数字、除零、缺字段、工具失败、需要查询后才知道的信息、避免重复执行的事务状态。只训练顺利成功的轨迹，模型不会学到真实环境中的恢复。

正确性过滤要在**训练切分**上执行。过滤后保存通过率和题目难度，避免只剩下很简单的样本。教师模型提供的示范必须记录教师版本、prompt、生成预算和标签可靠性；外部教师引入的新知识属于蒸馏成本。

### 14.6 训练与超参

固定预训练 checkpoint，使用完整参数微调做主对照；LoRA 可作为单独效率实验。学习率一般比 pretrain 小，但具体范围由 pilot 决定，例如教学起点扫描 `1e-5,3e-5,1e-4`，训练 1～3 epoch 并检查 holdout，而不是照搬某个 8B 模型的 recipe。

记录 assistant supervised tokens、全部 processed tokens、样本数、epochs、序列长度、真实更新数和全局 batch。监督 tokens 变少时，只比较 optimizer steps 会误导。

对短任务避免把长回答强塞到固定最大长度。长度 bucketing／packing 可提升效率，但 packing 必须保持样本边界、角色与 loss mask 正确。

### 14.7 SFT 验收

测格式率、任务成功、错误类型、跨模板／数字范围泛化、语言建模保持、回答长度与 stop 行为。人工抽查是否只是复读示范、模板记忆或过早 EOS。

重要消融：数据量、正确示范过滤、短／长示范、response-only／all-token loss、成功轨迹／含恢复轨迹、学习率、epoch。先过拟合小样本确认 mask，再扩大。

<a id="chapter-15"></a>
## 15. DPO：从偏好对学习

### 15.1 数据与目标

每个样本包含同一 prompt x、优选回答 y+、劣选回答 y−。冻结参考策略 `π_ref`，训练策略 `πθ`。标准 DPO：

$$
\mathcal L_{DPO}=-\mathbb E\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{ref}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{ref}(y^-\mid x)}\right]\right).
$$

序列 log-prob 是所有 response tokens 条件 log-prob 的**和**，通常包含终止 token，并排除 prompt 与 padding。DPO 不需要每个训练 step 在线 rollout，但偏好数据的生成和评审仍可能很昂贵。[R26](12_模板公式与参考.md#r26)

### 15.2 为什么有 reference

在 KL 正则化的奖励优化中，最优策略满足 `π*(y|x)∝π_ref(y|x)exp(r(x,y)/β)`。把这种关系代入成对偏好模型，就得到可以直接训练 policy 的目标。

reference 是相对概率基准，不是“把正负回答的 loss 都拟合到参考模型”。β 同时影响正则化解释和训练梯度尺度；β 大小不能只用“越大训练越强”一句话概括。

### 15.3 可运行的核心 loss

与第 14 章的 `response_log_probs` 配合，先得到各回答的序列 log-prob：

```python
def sequence_response_logp(model, ids, response_mask):
    lp, mask = response_log_probs(model, ids, response_mask)
    return (lp * mask).sum(-1), mask.sum(-1)


def dpo_loss(policy_chosen, policy_rejected,
             reference_chosen, reference_rejected, beta=0.1):
    delta = ((policy_chosen - reference_chosen)
             - (policy_rejected - reference_rejected))
    z = beta * delta
    loss = torch.logaddexp(torch.zeros_like(z), -z).mean()
    return loss, z.detach()
```

reference 用 `eval()` 和 `no_grad()`，参数 `requires_grad=False`；不能在训练中自动跟着 policy 更新。若 policy 与 reference 相同且没有 dropout／数值差异，初始 z≈0、loss≈log 2，可作为基本检查。

实际 step 的合同：

```text
policy chosen/rejected forward -> sequence log-probs with gradients
reference chosen/rejected forward -> frozen sequence log-probs
DPO objective -> gradient accumulation -> clip -> optimizer update
log chosen/rejected NLL, preference margin, length, eval quality
```

### 15.4 偏好对如何产生

可验证任务中，对同一 prompt 采样多个回答，用规则分成正确／错误，构造正负对。两者都正确、都错误或 label 不确定时，不应强行伪造高可信的优劣关系；可以跳过、标 tie 或建立单独软标签目标。

通用任务用 rubric／人工／judge 判断。至少记录：判官版本、规则、AB 顺序、长度、标签置信度、任务族。优选回答更长这一偏差如果普遍存在，DPO 会学习到长度相关行为。

离策略偏好来自其他模型；在策略偏好来自当前／近期 policy。二者覆盖和难度不同。迭代 DPO 可重新采样再训练，但每轮都要独立记录数据分布与费用。

### 15.5 Standard DPO 与长度归一化变体

标准 DPO 用序列和。长度归一化变体用 `Σ logp / response_length`，目标已发生变化，β 的有效尺度也变了；不能把它叫成原样的标准 DPO。

Tülu 3 在其特定模型／数据上研究了长度归一化 DPO，并将 SFT、偏好训练、RLVR 组成完整后训练 recipe。[R27](12_模板公式与参考.md#r27) 自己的 0.1B 实验应同时比较 standard／length-normalized、长度匹配的数据和独立长度指标。

### 15.6 缓存 reference log-probs

reference 固定时可以提前计算序列 log-prob，节省在线显存和算力。cache key 必须包含 reference hash、Tokenizer hash、完整 token IDs、response mask、position／attention 与 renderer 版本。

若训练时动态截断、重新 pack 或改写 prompt，原 cache 可能失效。只有相同条件序列的 frozen log-prob 才能复用。

### 15.7 常见失败

| 现象 | 可能原因 | 检查 |
| --- | --- | --- |
| 初始 loss 远离 log 2 | reference／policy 不同，mask 或 dropout 不同 | 逐 token log-prob、权重 hash |
| preference accuracy 上升，任务率下降 | 标签偏差、长度投机、离策略分布 | oracle、chosen NLL、长度匹配评测 |
| 两边 log-prob 都下降 | 目标只要求相对 margin | 看绝对 chosen NLL 与生成质量 |
| loss 快速饱和 | β／LR、容易偏好对、过拟合 | 更难／在策略对、开发集 |
| 参数几乎不更新 | 冻结错对象、detach policy、重复 norm | gradient norms、requires_grad |
| 显存比预期高 | 两模型、激活、双回答 batch | reference cache、micro-batch |

### 15.8 DPO 验收

训练之外使用新 prompt、新候选与独立 verifier／judge；报告 preference accuracy、真实成功、格式、长度、KL／NLL 保持、总生成与判官成本。训练偏好准确率只是优化诊断，不是最终任务能力。
