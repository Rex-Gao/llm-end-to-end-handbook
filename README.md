# 大模型全链路知识手册

从原始语料出发，亲手实现 Tokenizer 与约 0.1B Transformer，完成预训练、性能优化、后训练和可验证 Agent 实验。手册把算法推导、参考代码、实验设计、性能测量和验收标准放在同一条学习路线上。

**[阅读完整手册](大模型全链路知识手册.md)** · **[查看分册目录](#分册目录)** · **[查看实战安排](大模型全链路知识手册.md#chapter-23)** · **[查看参考资料](handbook_parts/12_模板公式与参考.md#appendix-c)**

## 内容与交付

手册包含 25 个主章节、3 个附录和 40 项参考资料，按学习顺序拆为 12 册。完整手册适合连续阅读与全文查找；分册适合按阶段学习、讨论和记录实验。

| 阶段 | 主要内容 | 建议形成的实验产物 |
| --- | --- | --- |
| 工程与数据 | 环境、复现清单、raw dump 解析、质量过滤、去重、切分与污染检查 | 数据清单、清洗报告、数据版本与 shards |
| 从零实现 | Byte-level BPE、RMSNorm、RoPE、causal Attention、SwiGLU、AdamW | Tokenizer、模型、优化器及正确性检查 |
| 预训练与推理 | next-token loss、混合精度、累积、checkpoint、采样、KV Cache | 训练日志、checkpoint、生成样例与推理基线 |
| 系统优化 | Triton Attention 前向／反向、profiling、DDP／FSDP、训练与推理扩展 | 数值对照、性能表、显存分析和多卡收益报告 |
| 规模与评测 | FLOPs 与预算、Scaling Law 拟合、外推、统计和受控消融 | 拟合数据、置信范围、外推验证及评测协议 |
| 后训练 | SFT、DPO、RLVR，以及同一基座、同一任务与预算下的对照 | 后训练模型、奖励审计和对照实验报告 |
| Agent 与闭环 | 可验证环境、harness、长程轨迹、信用分配、self-judge | 环境卡、trace、Agent 评测与闭环消融 |
| 发布与维护 | 模型卡、服务、观测、回滚、验收和排错 | 发布包、回归记录及实验总报告 |

默认 scratch 模型的独立参数量为 **97,536,768**，采用 16,384 词表、768 hidden、12 层、12 个 Attention heads、2,048 FFN hidden 与 tied embeddings。该配置用于教学实验；正文给出参数量推导、显存预算与训练计划。

## 分册目录

| 分册 | 对应章节 | 学习重点 |
| --- | --- | --- |
| [01 · 导读与工程](handbook_parts/01_导读与工程.md) | 0–2 | 目标、先修知识、符号、工程目录与复现协议 |
| [02 · 数据与分词](handbook_parts/02_数据与分词.md) | 3–4 | raw dump、过滤／去重／污染、Byte-level BPE、训练 shards |
| [03 · 模型与预训练](handbook_parts/03_模型与预训练.md) | 5–6 | 约 0.1B Transformer、Attention、优化器与训练循环 |
| [04 · 资源推理与测量](handbook_parts/04_资源推理与测量.md) | 7–9 | FLOPs、显存、KV Cache、采样、吞吐与延迟 |
| [05 · Triton 与多卡](handbook_parts/05_Triton与多卡.md) | 10–11 | tiled Attention、前向／反向、多卡训练和推理 |
| [06 · Scaling 与评测](handbook_parts/06_Scaling与评测.md) | 12–13 | Scaling Law、拟合／外推、统计、公平评测与消融 |
| [07 · SFT 与 DPO](handbook_parts/07_SFT与DPO.md) | 14–15 | chat template、response-only loss、偏好对和 DPO |
| [08 · RLVR 与对照](handbook_parts/08_RLVR与对照.md) | 16–17 | verifier、策略梯度、组优势、KL 与共享基座对照 |
| [09 · 环境与 Harness](handbook_parts/09_环境与Harness.md) | 18–19 | 任务与奖励、可重放环境、工具协议、状态机和 trace |
| [10 · 长程 Agent 与 Self-judge](handbook_parts/10_长程Agent与SelfJudge.md) | 20–21 | 轨迹概率、信用分配、异步 rollout、judge 校准与闭环 |
| [11 · 发布实战与排错](handbook_parts/11_发布实战与排错.md) | 22–24 | 发布、回滚、阶段安排、练习、验收与排错 |
| [12 · 模板公式与参考](handbook_parts/12_模板公式与参考.md) | 附录 A–C | 实验记录、模型／环境卡、公式、术语与一手资料 |

各分册提供上一册／下一册入口。章节引用和参考资料链接可直接在分册之间跳转；完整手册保留内部目录导航。

## 推荐阅读路线

首次学习建议依次阅读 01–12，同时完成每个阶段的最小实验。先让数据、Tokenizer、模型和训练循环在小规模配置下通过正确性检查，再扩大参数量、序列长度和训练预算。

- **模型实现路线：** 01 → 02 → 03 → 04，完成数据与分词、默认模型、优化器、训练和缓存推理。
- **系统与规模路线：** 04 → 05 → 06，建立基线、实现 Triton、测量多卡效率，并用真实实验点拟合 Scaling Law。
- **后训练路线：** 06 → 07 → 08，固定任务、数据切分、基座与预算，对比 SFT、DPO、RLVR。
- **Agent 路线：** 08 → 09 → 10，先证明环境和 verifier 可靠，再训练长程行为并评估 self-judge。
- **交付路线：** 11 → 12，整理实验结果、失败案例、模型卡、环境卡和发布验收。

路线之间存在依赖。性能优化以数值正确为前提；后训练依赖可信评测；Agent 强化学习依赖可验证环境；自我改进的收益应由独立任务集确认。

## 如何使用代码与实验方案

本仓库交付 Markdown 教学文档。章节中的代码分别承担三种作用：

| 类型 | 示例 | 使用方式 |
| --- | --- | --- |
| 独立参考实现 | BPE、Transformer、AdamW、采样、缓存 Attention、SFT／DPO loss、verifier、Scaling 拟合 | 抽取到实验项目，补齐入口与数据，并先运行对应正确性检查 |
| 依赖目标运行环境的教学实现 | Triton Attention、DDP 梯度累积、FSDP 配置 | 锁定软件组合，在目标环境验证编译、数值、更新等价性与性能 |
| 模块合同与伪代码 | 完整训练流程、rollout／harness、发布协议 | 按合同补齐存储、任务接口、调度、异常恢复和日志 |

正文中的预算、阈值、课程进度与默认配置属于教学方案。论文、框架和官方接口的出处集中在附录 C；真实训练质量、吞吐、多卡收益和 Agent 成功率应由实际实验产生。合成数据拟合用于检查公式和拟合程序，不能直接替代真实 Scaling 实验。

## 先修知识与实验准备

读者需要掌握 Python、张量计算和自动微分，理解基本线性代数、概率、交叉熵与优化器。第 1 章统一符号并补充关键数学知识。

算法示例主要使用 Python、NumPy、PyTorch；Scaling 拟合使用 SciPy。Triton、多卡训练和推理服务按各章要求准备兼容的 CUDA 软件组合。正文 shell 示例采用 Linux／WSL2 Bash，具体安装版本由官方支持矩阵决定。

建议把数据处理、训练和 rollout 依赖分别锁定。每次实验保存数据／Tokenizer／模型版本、配置、seed、代码 commit、依赖清单、预算与原始结果。第 2 章提供工程目录和运行清单，第 7 章说明资源预算，第 23 章给出分阶段安排。

## 验收原则

1. **先检查正确性。** 验证 Tokenizer round-trip、特殊 token、参数量、causal mask、Attention 梯度、优化器与 KV Cache 等价性。
2. **再跑最小闭环。** 用小模型、小样本检查过拟合、checkpoint 恢复、数据消费与端到端日志。
3. **统一测量协议。** 性能比较固定 shape、batch、精度、预热和计时口径，并保留原始样本。
4. **公平比较后训练。** 共享基座、任务切分与预算，同时报告质量、长度、格式、成本和失败类型。
5. **独立评估 Agent 与 judge。** 分开记录奖励和真实任务正确率，审计 verifier，并冻结评测任务与 judge 版本。
6. **整理证据后发布。** 每项结论对应配置、日志、评测或原始测量；失败案例和适用范围随结果一起保留。

完整检查表见 [第 23.6 节](大模型全链路知识手册.md#chapter-23)，代码验证口径见 [附录 A](handbook_parts/12_模板公式与参考.md#appendix-a)，常见错误见 [第 24 章](大模型全链路知识手册.md#chapter-24)。

## 仓库结构

```text
.
├── README.md
├── 大模型全链路知识手册.md
└── handbook_parts/
    ├── 01_导读与工程.md
    ├── 02_数据与分词.md
    ├── 03_模型与预训练.md
    ├── 04_资源推理与测量.md
    ├── 05_Triton与多卡.md
    ├── 06_Scaling与评测.md
    ├── 07_SFT与DPO.md
    ├── 08_RLVR与对照.md
    ├── 09_环境与Harness.md
    ├── 10_长程Agent与SelfJudge.md
    ├── 11_发布实战与排错.md
    └── 12_模板公式与参考.md
```

维护内容时，以完整手册为主稿同步分册，检查章节锚点、参考资料与上下册导航，保证两种阅读方式的正文一致。

## 参考资料与反馈

附录 C 收录 Stanford CS336、PyTorch／Triton 官方资料、数据工程方法、FlashAttention、Scaling Law、DPO／RLVR、Agent 环境与 self-judge 等论文和项目，并说明各条资料的用途与边界。接口和默认行为可能随版本变化，实验应固定具体版本或 commit。

欢迎通过 Issue 或 Pull Request 提交勘误、公式推导、代码修正和实验结果。报告问题时，请附章节编号、复现步骤、预期／实际行为与必要的配置；提交性能或质量结论时，请同时提供测量协议和对照设置。
