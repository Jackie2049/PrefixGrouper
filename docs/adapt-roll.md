# PrefixGrouper 适配 ROLL：研究与开发准备

> 状态：调研阶段。本文基于 PrefixGrouper `roll-prefixgrouper` 分支，以及 `dependency/ROLL` 中的 ROLL `origin/main` 快照（`78c8c7d`）编写。

## 1、研究分析

### 1.1 目标与边界

本适配的目标是为 ROLL 的 **固定 prompt、多 completion 的 GRPO/RLVR** 训练路径提供 PrefixGrouper 的 shared-prefix forward：每个 prompt 在每层只计算一次，随后让同组 completion 的 suffix attention 复用该 prompt 的 K/V。它与推理侧 prefix caching、以及面向任意前缀树的 PrefixSharing 不同。

本项目仅考虑 ROLL + FSDP2 + PrefixGrouper：第一阶段只支持文本 causal LM 的 actor 路径、单卡 FSDP2 与固定 prompt 的 GRPO/RLVR。动态 batching、sequence packing、多模态、LoRA/reference 与 agentic 任意长度轨迹均不在当前调研、设计或开发范围内。

### 1.2 PrefixGrouper 的现有契约

PrefixGrouper 以 `group_info = [[prefix_len, suffix_1_len, ...], ...]` 描述一个 micro-batch。`from_ungrouped_masks()` 可由独立 prompt mask、completion mask 与每个 prompt 的 group size 自动生成该结构。

关键执行链如下：

1. `concat_input(prefix, prefix_mask, suffix, suffix_mask)` 将 `B` 条 prompt 与 `sum(G_i)` 条 completion 压缩为 `B` 条 grouped sequence。
2. 模型 forward 额外接收 `prefix_grouper`。
3. 每层 attention 将 Q/K/V 拆为 prefix 和 suffix：prefix 做一次 attention；suffix Q 对「重复后的 prefix K/V + 自身 suffix K/V」做 attention；再 group 回原输出布局。
4. `split_output(..., include_prefix_last=1)` 还原 completion logits/mask。prefix 最后一个 token 的 logits 用于预测每个 completion 的首 token，因此需复制到每个 suffix 输出的首位。
5. 后续 GRPO loss / backward 使用恢复后的 suffix 输出，语义与未压缩 baseline 一致。

当前实现是 PyTorch autograd 运算：`GroupFunction`/`UngroupFunction` 负责 gather/scatter，attention wrapper 假设 Q/K/V 为 dense `[B, H, S, D]`，并以 Transformers `AttentionInterface` 注册 `prefix_grouper_attention`。其直接可用模型路径是 HuggingFace Transformers；这与 ROLL 的 FSDP2 strategy 是本项目唯一讨论的后端组合。

### 1.3 ROLL 的相关数据和 rollout 语义

ROLL 的 `RLVRConfig.num_return_sequences_in_group` 会写入 actor inference 的 `generating_args.num_return_sequences`。vLLM/SGLang 返回 `n` 个 completion 后，`concatenate_input_and_output()` 把同一 prompt `repeat(n)`，形成常规完整样本：

```text
[P][R1]
[P][R2]
...
[P][RG]
```

这正是 PrefixGrouper 所需的原始信息，但 ROLL 当前未保留一个显式的「prompt -> completion rows」训练接口：`prompt_id`、`group_ids` 等 metadata 在不同路径中含义不同，不能直接作为 PrefixGrouper group contract。适配层需要在 rollout 完成后、任何重排前，以 token-level prompt 边界及 group membership 构造稳定的 group map。

ROLL 的 actor loss 使用完整 `input_ids` 与 `response_mask[:, 1:]` 计算 logprob、KL、ratio、entropy 和 GRPO/PPO loss。因此适配后必须在进入现有 loss 之前把 grouped logits 还原为原有 `[sum(G_i), S, V]`（或等价的 logprob/entropy）语义；不能改变现有算法 worker 的输入契约。

### 1.4 ROLL 的训练入口与候选落点

#### FSDP2 路径

`roll/distributed/strategy/fsdp2_strategy.py` 的 forward/train loop 更直接调用 `_fsdp2_forward(input_ids, attention_mask, position_ids, forward_args)`。若模型使用支持 `AttentionInterface` 的 Transformers 版本，PrefixGrouper 的 attention registration 更接近可复用；但 ROLL 的 model wrapper 仍须透传 `prefix_grouper`，并验证 FSDP2、自动混精与既有梯度同步行为。

#### Pipeline/worker 路径

`roll/pipeline/base_worker.py` 统一调用 strategy 的 `compute_log_probs()` 和 `train_step()`；`roll/pipeline/rlvr/actor_worker.py` 在 loss 中用 `op_compute_log_probs()` 计算 token logprob。因此推荐将数据重组、模型调用和输出恢复封装在 strategy 内，避免向 actor loss 暴露 PrefixGrouper 专用张量。

### 1.5 调度与并行限制

1. `batch_balance()` 按单条序列长度进行 DP 负载均衡并重排。它会拆散同一 prompt 的 completion；PrefixGrouper 要求同一组位于同一 DP rank 和同一 micro-batch。因此需要 group-aware 的分区与装箱，而不是在当前 `batch_balance()` 之后临时 regroup。
2. 当前 dynamic batching 与 sequence packing 也会独立重排/切分样本。本项目不支持它们；适配前必须通过兼容性 gate 回退原路径。

### 1.6 关键不变量

- 每个 completion 的 token logprob、entropy、KL、policy loss 与 baseline 对齐；
- `include_prefix_last=1` 后，第一个 completion token 的预测来自共享 prefix 的最后一个 hidden state；
- response mask、final response mask、advantages、old/ref/infer logprobs 必须按原 completion row 和 token 位置恢复；
- 组内顺序、group reward/advantage 语义及 DP 样本数统计保持不变；
- PrefixGrouper 关闭或 batch 不符合条件时严格回退到 ROLL 原路径。

## 2、方案设计

### 2.1 推荐分层

在 PrefixGrouper 仓库新增 `integrations/roll/`，保持核心 `src/prefix_grouper/` 框架无关：

```text
integrations/roll/
  config.py          # 读取 ROLL 配置与兼容检查
  grouping.py        # prompt/completion 映射、group-aware partition
  batch.py           # 原 batch <-> PrefixGrouper grouped batch
  outputs.py         # logits/logprob/entropy/mask 恢复
  fsdp2.py           # HF/FSDP2 adapter
```

ROLL 侧长期只应拥有小型 hook/config；PrefixGrouper 算法、数据变换、模型后端适配和精度测试保留在本仓库，避免复制实现。

### 2.2 MVP：FSDP2 + 文本 GRPO

实现 FSDP2/HF 文本模型 MVP：

1. 增加 `prefix_grouper.enabled`、`min_group_size`、`backend`、`fallback_on_ineligible_batch` 配置。
2. 在 RLVR rollout 输出后生成显式 `prefix_group_id`、`prompt_row_index`、`completion_index` 与 prompt token length。
3. 在 DP 分发前把一个完整 group 作为原子单元装箱；第一版要求 group 不跨 DP/micro-batch。
4. FSDP2 strategy 将完整 rows 变换为 PrefixGrouper grouped input，并把 `prefix_grouper` 透传到 model/attention。
5. forward 后恢复 completion-row logits 与 masks，使既有 `ActorWorker.loss_func()` 不变。
6. 不符合模型/并行/输入条件时走 baseline。

### 2.3 ROLL 上游 PR 边界

建议拆分为：

1. **ROLL PR A：数据契约与 group-aware 分发 hook**。新增稳定 metadata 和可选 group-preserving partition 接口，不引入 PrefixGrouper 依赖。
2. **PrefixGrouper：ROLL adapter PoC**。提供外部安装、版本检查、FSDP2 精度测试。
3. **ROLL PR B：可选集成**。由显式配置启用 adapter；默认行为不变。

## 3、测试验证

### 3.1 单元测试（CPU）

- group map：固定 group size、可变 group size、混合 prompt、无有效 group；
- batch transform：input_ids、attention mask、position ids、response/final_response mask、advantages、old/ref/infer logprobs 的 gather/scatter；
- prefix-last 边界：completion 第一个 token 与不同 response 长度；
- fallback：不满足条件时 byte-for-byte 保持 baseline batch；
- group-aware partition：不跨 DP/micro-batch、可解释 workload 指标。

### 3.2 数值等价测试（GPU）

对同一固定 rollout 比较 PrefixGrouper off/on：

- forward logits、token logprob、entropy；
- KL、ratio、actor loss；
- 参数梯度、grad norm、optimizer step 后参数；
- group size 2/4/8，短/长 prompt，变长 completion；
- BF16/FP32 容忍度需逐项记录。

### 3.3 集成测试

1. 单卡 FSDP2 文本 GRPO smoke test；
2. DP=2 group-preserving dispatch test；
3. PrefixGrouper 关闭、无共享、配置不兼容三种回退；
4. FSDP2 自动混精及不同 group size 的兼容性测试。

### 3.4 性能验证

报告固定模型、prompt 长度、completion 长度、group size、GPU、dtype 和并行度下的：

- actor forward/backward 时间；
- end-to-end train step 时间；
- peak allocated/reserved memory；
- group transform 与 restore 开销；
- 有效 prefix token ratio 与理论/实测加速比。

## 4、开发计划

### Phase 0：协议与基线

- [ ] 固定 ROLL 与 PrefixGrouper 版本；
- [ ] 选定一个文本 GRPO 示例和固定 rollout fixture；
- [ ] 记录 baseline logits/loss/gradient/perf；
- [ ] 定义 ROLL group metadata 和 eligibility 规则。

### Phase 1：数据适配与测试

- [ ] 实现 group map 与完整 batch transform/restore；
- [ ] 添加 CPU 单测和不变量检查；
- [ ] 实现 group-aware partition 原型；
- [ ] 提交 ROLL 数据 hook RFC/小 PR。

### Phase 2：FSDP2 PoC

- [ ] 增加 Transformers attention registration/透传；
- [ ] 打通单卡文本 GRPO actor forward/backward；
- [ ] 完成 on/off 数值等价和显存/吞吐测量；
- [ ] 加入兼容性 gate 与 baseline fallback。

### Phase 3：上游化

- [ ] ROLL 可选集成 PR；
- [ ] 补齐 FSDP2 的安装、配置与回退说明；
- [ ] 固化兼容性矩阵和复现实验脚本。

## 5、当前结论

1. ROLL 已有 GRPO group rollout 所需的基础数据，但训练前会复制完整 prompt，未实现 PrefixGrouper。
2. PrefixGrouper 的核心数据/attention/output 恢复语义可复用，且其 Transformers B/H/S/D attention adapter 与 ROLL FSDP2/HF 模型组合存在直接的复用路径。
3. 最低风险的实现顺序是：先完成 ROLL 数据契约，再完成 FSDP2/HF 文本 PoC 与数值等价验证。
4. 适配必须把完整 group 作为 DP/micro-batch 调度原子，否则共享前缀不会命中。
5. 对 ROLL 的长期贡献应是可选、可回退的 group-aware hook；PrefixGrouper 算法实现应保持在本仓库。

## 6、遗留问题

- ROLL rollout batch 中何处最可靠地保留 prompt/completion 边界及稳定 group relation？现有 `prompt_id`/`group_ids` 是否足以覆盖同步、异步和 agentic path？
- FSDP2 所用 Transformers 版本和模型 wrapper 是否能无侵入地透传 `prefix_grouper`，或需 ROLL 添加 model-forward hook？
- group-aware DP 负载均衡如何同时保证组完整性和长序列负载均衡？
- PrefixGrouper 为 MIT、ROLL 为 Apache-2.0；两者通常可兼容，但上游集成时需保留 MIT notice，并确认是否采用可选依赖而非复制源码。
