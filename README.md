# GRPO_fsy

这个仓库整理了我学习和手写 GRPO（Group Relative Policy Optimization）的笔记与最小 PyTorch 实现。重点不是复现完整工业级 RLHF 框架，而是把 GRPO 的核心训练逻辑拆开：同一个 prompt 采样多条回答，在组内比较奖励高低，用相对优势更新模型。

一句话概括：

```text
GRPO = 对同一个问题生成多条回答，让回答之间互相当 baseline，
再把高于组内平均的回答概率调高，把低于组内平均的回答概率调低。
```

## 仓库内容

```text
.
├── GRPO.md                     # 手撕 GRPO 主笔记：公式、shape、训练流程、易错点
├── infer_py讲解.md              # Search-R1 infer.py 推理流程讲解
├── grpo_from_scratch/
│   ├── train.py                # 最小版 GRPO trainer
│   ├── reward_func.py          # 规则奖励函数
│   ├── test.py                 # OpenAI-compatible 接口调用示例
│   └── grpo_loss.png           # GRPO/PPO loss 示意图
└── README.md
```

## GRPO 核心流程

最小版 GRPO 训练可以拆成 7 步：

1. 取一个 prompt。
2. 把同一个 prompt 复制 `num_generations` 份。
3. 用当前策略模型生成多条 response。
4. 用奖励函数给每条 response 打分。
5. 在同一个 group 内标准化 reward，得到每条 response 的 advantage。
6. 重新前向模型，计算当前 `log_prob` 与采样时 `old_log_prob` 的 ratio。
7. 使用 PPO clipped objective 更新策略模型。

核心公式：

```text
advantage_i = (reward_i - mean(rewards)) / (std(rewards) + eps)
ratio = exp(log_prob_current - log_prob_old)
loss = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

和普通 PPO 相比，这份实现没有 value model；它用同一个 prompt 下多条回答的组内均值和标准差来构造相对优势。

## `train.py` 从头到尾梳理

主要实现位于 [`grpo_from_scratch/train.py`](./grpo_from_scratch/train.py)。代码可以按下面几层理解：

| 模块 | 作用 |
|---|---|
| `GSM8KDataset` | 读取中文 GSM8K 数据，返回 `prompt` 和 `answer` |
| `Samples` | 保存同一个 prompt 下的一组生成样本 |
| `GRPOArguments` | 配置学习率、生成条数、序列长度、KL 系数等超参 |
| `GRPOTrainer.__init__` | 初始化策略模型、可选 reference model、奖励函数、优化器和经验 buffer |
| `GRPOTrainer.generate_samples` | 对同一个 prompt 采样多条回答，得到 `prompt + response`、mask 等 |
| `GRPOTrainer.generate_experiences` | 计算 reward、advantage、old log prob、reference log prob |
| `GRPOTrainer.get_action_log_probs` | 从模型输出里取出 response token 的 log probability |
| `GRPOTrainer.compute_loss` | 计算 GRPO/PPO 风格的 clipped policy loss |
| `GRPOTrainer.train_step` | 执行一次 backward，并在满足梯度累积条件时更新参数 |
| `GRPOTrainer.train` | 外层训练循环：采样经验、缓存经验、多轮复用、保存 checkpoint |
| `__main__` | 加载模型、tokenizer、数据集、奖励函数并启动训练 |

### 1. 数据集与配置

`GSM8KDataset` 是一个 PyTorch `Dataset`。初始化时用 `datasets.load_dataset(data_path)` 加载数据集，`__getitem__` 返回：

```python
{"prompt": sample["question_zh-cn"], "answer": sample["answer_only"]}
```

`GRPOArguments` 里保存训练超参：

| 变量 | 含义 |
|---|---|
| `output_dir` | 模型和 checkpoint 输出目录 |
| `device` | `cuda` 或 `cpu` |
| `lr` | Adam 学习率 |
| `save_steps` | 每多少次 `update_steps` 保存一次 checkpoint |
| `epoch` | 数据集训练轮数 |
| `num_generations` | 每个 prompt 生成多少条 response，也就是 group size |
| `max_prompt_length` | prompt token 最大长度 |
| `max_generate_length` | response token 最大长度 |
| `reward_weights` | 多个奖励函数的权重 |
| `beta` | KL 惩罚系数；为 0 时不启用 reference model |
| `clip_eps` | PPO ratio clipping 范围 |
| `gradient_accumulation_steps` | 梯度累积步数 |
| `num_iterations` | 同一批 rollout 经验重复训练几轮 |
| `batch_size` | dataloader 每次取多少个 prompt |

### 2. Trainer 初始化

`GRPOTrainer.__init__` 主要做这些事情：

1. 加载或接收策略模型 `self.model`，并移动到 `args.device`。
2. 如果 `beta != 0`，复制一份 `ref_model`，用于 KL 惩罚；这份模型只 eval，不训练。
3. 初始化 tokenizer，并设置 `padding_side = "left"`。
4. 处理奖励函数：如果 `reward_funcs` 里是字符串，就按奖励模型路径加载 `AutoModelForSequenceClassification`；如果是普通 Python 函数，就直接使用。
5. 为奖励模型补齐 `reward_tokenizer` 和 `pad_token`。
6. 创建 Adam 优化器。
7. 创建 `input_buffer`，长度等于 `gradient_accumulation_steps`，用于缓存短期 rollout 经验。

这里的 `input_buffer` 不是离线 RL 的大数据池，它只是短时间保存刚采样出来的几批经验，用于梯度累积和 `num_iterations` 次复用。

### 3. 采样样本：`generate_samples`

`generate_samples(inputs)` 的目标是：对 batch 内每个 prompt，生成一个 group 的回答。

核心流程：

```text
batch prompt
  -> 对每个 prompt 应用 chat template
  -> 复制 num_generations 份
  -> model.generate 生成多条 response
  -> 整理到固定长度 max_prompt_length + max_generate_length
  -> 构造 attention_mask / response_ids / action_mask
  -> 打包成 Samples
```

关键变量：

| 变量 | shape | 含义 |
|---|---|---|
| `prompt_ids` | `[num_generations, max_prompt_length]` | 同一个 prompt 复制后的 token ids |
| `prompt_response_ids` | `[num_generations, max_prompt_length + max_generate_length]` | `prompt + response` 的完整 token 序列 |
| `attention_mask` | `[num_generations, max_prompt_length + max_generate_length]` | 给 Transformer 用，标记非 pad token |
| `response_ids` | `[num_generations, max_generate_length]` | 只保留模型新生成的 response 部分 |
| `action_mask` | `[num_generations, max_generate_length]` | 给 loss 用，标记参与策略梯度的 response token |
| `response_length` | `[num_generations]` | 每条 response 的有效 token 数 |

`attention_mask` 和 `action_mask` 的区别很关键：

```text
attention_mask:
  作用范围是 prompt + response
  用于 Transformer 前向计算
  主要 mask 掉 pad token

action_mask:
  作用范围只在 response
  用于 GRPO/PPO loss
  mask 掉 eos 和 pad token
```

decoder-only Transformer 里的 causal mask 通常由模型内部自动处理；这里传入的 `attention_mask` 主要是 padding mask。

### 4. 生成经验：`generate_experiences`

“经验”不是原始数据集样本，而是当前策略模型和奖励函数交互后得到的训练数据。它包括：

```python
{
    "prompt_response_ids": ...,
    "attention_mask": ...,
    "action_mask": ...,
    "old_action_log_probs": ...,
    "ref_action_log_probs": ...,
    "advantages": ...,
}
```

完整流程：

```text
generate_samples
  -> 得到每个 prompt 的 num_generations 条回答
  -> 用当前策略模型计算 old_action_log_probs
  -> 如果 beta != 0，用 ref_model 计算 ref_action_log_probs
  -> 解码 response_ids 得到 response_texts
  -> 用奖励函数或奖励模型打分
  -> 多个 reward 按 reward_weights 加权求和
  -> 在 group 内标准化 reward 得到 advantage
  -> 把多个 prompt 的 group 在 batch 维拼接
```

`old_action_log_probs` 很重要。它表示采样这些 response 时，旧策略对每个 response token 的 log probability。后面训练时模型参数已经变化，所以会重新计算当前 `action_log_probs`，再得到 PPO ratio：

```text
ratio = exp(action_log_probs - old_action_log_probs)
```

`advantages` 是 response 级别的，不是 token 级别的。假设一个 prompt 生成 4 条回答：

```text
rewards = [0.2, 1.0, 0.5, 0.0]
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

reward 高于组内平均，advantage 为正；低于组内平均，advantage 为负。

### 5. 取 token log probability：`get_action_log_probs`

这个函数把模型对完整序列的输出转换成 response token 的 log probability。

输入：

```text
input_ids = prompt + response
```

模型输出：

```text
logits.shape = [batch, seq_len, vocab_size]
```

decoder-only 语言模型中：

```text
logits[:, t, :] 用来预测 input_ids[:, t + 1]
```

所以代码先做：

```python
log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
```

去掉最后一个位置，因为最后一个 logits 没有下一个真实 label。然后用：

```python
log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
```

从整个词表分布中取出“真实下一个 token”的 log probability。最后：

```python
action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
```

只取最后 `num_actions` 个位置，也就是 response 部分的 token log probability。

### 6. 损失函数：`compute_loss`

`compute_loss` 是 GRPO/PPO 更新的核心。

首先重新用当前模型计算 response token 的 `action_log_probs`。然后取旧策略概率：

```python
old_action_log_probs = (
    inputs["old_action_log_probs"]
    if self.args.num_iterations > 1
    else action_log_probs.detach()
)
```

如果同一批经验训练多轮，就使用采样时缓存的旧概率；如果只训练一轮，这里用当前概率 detach 后当旧概率。

PPO ratio：

```python
coef_1 = torch.exp(action_log_probs - old_action_log_probs)
coef_2 = torch.clamp(coef_1, 1 - clip_eps, 1 + clip_eps)
```

`advantages` 是 `[batch]`，而 `coef_1` 是 `[batch, num_actions]`，所以需要：

```python
advantages.unsqueeze(1)
```

把 `[batch]` 变成 `[batch, 1]`，然后广播到每个 token：

```text
同一条 response 的所有 token 都乘同一个 response-level advantage
```

最终 token loss：

```python
per_token_loss1 = coef_1 * advantages.unsqueeze(1)
per_token_loss2 = coef_2 * advantages.unsqueeze(1)
per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
per_token_loss = per_token_loss * action_mask
```

直观含义：

```text
advantage > 0 的 response：提高这些 token 的概率
advantage < 0 的 response：降低这些 token 的概率
clip：限制当前策略不要偏离 old policy 太多
action_mask：只让有效 response token 参与 loss
```

如果 `beta != 0`，还会加入 KL 惩罚：

```python
log_ratio = ref_action_log_probs - action_log_probs
k3 = log_ratio.exp() - 1 - log_ratio
per_token_loss = per_token_loss + beta * k3
```

最后先对每条 response 的有效 token 求平均，再对 batch 求平均：

```python
loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)
loss = loss.mean()
```

### 7. 单步训练：`train_step`

`train_step` 只负责一次训练动作：

```text
model.train()
  -> compute_loss
  -> loss / gradient_accumulation_steps
  -> backward
  -> 如果累计够步数：optimizer.step + optimizer.zero_grad
  -> 写 TensorBoard 和打印 loss
```

注意 `train_step` 不是完整训练循环。它不会采样 response，也不会计算 reward；它只消费 `generate_experiences` 已经生成好的经验。

### 8. 外层循环：`train`

`train` 是完整训练流程：

```text
for epoch:
  dataloader shuffle 取 batch
  每个 batch:
    generate_experiences(batch)
    放入 input_buffer
    如果攒够 gradient_accumulation_steps 个 batch:
      对同一批 buffer 经验重复训练 num_iterations 轮
      每轮里依次调用 train_step
      update_steps += 1
      定期保存 checkpoint
```

这就是“短期 on-policy rollout 复用”：

```text
当前模型采样一小批经验
  -> 用这批经验训练 num_iterations 轮
  -> 模型更新
  -> 再用更新后的模型采样下一小批经验
```

它不是完全离线 RL，因为经验不会长期固定；但也不是每 backward 一次都重新采样，因为 PPO/GRPO 会短期复用 rollout 来提高样本利用率。

### 9. 脚本入口

`if __name__ == "__main__"` 部分做启动训练的准备：

1. 设置 `CUDA_VISIBLE_DEVICES`。
2. 定义 `SYSTEM_PROMPT`，要求模型按 `<think>` 和 `<answer>` 格式输出。
3. 创建 `GRPOArguments`。
4. 创建 TensorBoard `SummaryWriter`。
5. 加载 Qwen tokenizer 和 causal LM。
6. 加载 GSM8K 中文数据集。
7. 创建 `GRPOTrainer`，传入规则奖励函数。
8. 调用 `trainer.train()` 开始训练。
9. 调用 `trainer.save_model()` 保存最终模型。

## 关键变量速查

| 变量 | 来自哪里 | 典型 shape | 用途 |
|---|---|---|---|
| `prompt` | 数据集 | Python 字符串 | 原始问题 |
| `answer` | 数据集 | Python 字符串 | 标准答案，给 reward 函数使用 |
| `input_text` | chat template | Python 字符串 | 加入 system/user 格式后的 prompt |
| `prompt_ids` | tokenizer | `[G, P]` | 同一个 prompt 复制 G 份后的 token ids |
| `prompt_response_ids` | `model.generate` | `[G, P + R]` | 完整序列，作为后续模型前向输入 |
| `response_ids` | 切片 | `[G, R]` | 只包含生成回答 |
| `attention_mask` | 非 pad 判断 | `[G, P + R]` | Transformer padding mask |
| `action_mask` | response 非 eos/pad 判断 | `[G, R]` | loss mask |
| `old_action_log_probs` | 采样后重新前向 | `[G, R]` | 旧策略 token log prob |
| `ref_action_log_probs` | reference model | `[G, R]` | KL 惩罚使用 |
| `rewards_per_func` | 奖励函数 | `[num_reward_funcs, G]` | 每个奖励函数对每条回答的分数 |
| `rewards` | 加权求和 | `[G]` | 每条 response 的最终奖励 |
| `advantages` | group 标准化 | `[G]` | response 级别优势 |
| `action_log_probs` | 当前模型重新前向 | `[batch * G, R]` | 当前策略 token log prob |
| `coef_1` | PPO ratio | `[batch * G, R]` | 未裁剪 ratio |
| `coef_2` | PPO clipped ratio | `[batch * G, R]` | 裁剪后 ratio |
| `per_token_loss` | loss 计算 | `[batch * G, R]` | token 级损失 |

这里 `G = num_generations`，`P = max_prompt_length`，`R = max_generate_length`。

## 技术关键点

- `attention_mask` 是 HuggingFace 模型前向的 padding mask，不是手写 Transformer 里的 causal mask。causal mask 通常由 decoder-only 模型内部自动构造。
- `action_mask` 是 RL loss mask，只对应 response 部分，用来排除 eos/pad 等不应训练的位置。
- `advantage` 是 response 级别的，所以训练时用 `advantages.unsqueeze(1)` 广播到该 response 的所有 token。
- `old_action_log_probs` 必须固定为采样时策略的概率，否则 PPO ratio 就失去“新旧策略比较”的意义。
- `num_iterations` 越大，旧 rollout 被复用越久，样本利用率更高，但策略偏离旧策略的风险也更大。
- `clip_eps` 限制 ratio 范围，缓解用旧 rollout 多轮训练带来的不稳定。
- `beta` 控制 KL 惩罚；`beta = 0` 时不使用 reference model。
- `reward_weights` 用来合并多个奖励函数，权重数量必须和奖励函数数量一致。
- `get_action_log_probs` 里的 `gather` 是从词表维度取“真实生成 token”的 log prob，而不是取最大概率 token。

## 易错点

- `generate(...)` 当前传了 `temperature/top_k/top_p`，但没有显式传 `do_sample=True`。在 HuggingFace 里如果不启用 sampling，可能走 greedy decoding，采样参数不生效或触发 warning。
- 如果 tokenizer 没有 `pad_token_id`，`attention_mask`、padding 和 generate 都可能出问题，通常需要显式设置 `tokenizer.pad_token = tokenizer.eos_token`。
- `action_mask` 的 dataclass 类型标注是 `BoolTensor`，但代码里实际转成了 `long`。这不一定影响乘法，但类型标注和实现不一致。
- `action_mask.sum(dim=1)` 如果为 0，会导致 loss 除零。比如模型一开始就生成 eos，或者 response 全是 pad，需要额外保护。
- `reward_func` 返回 `None` 会被替换为 `nan`，但后面没有处理 `nan` 扩散；一旦进入 `rewards.mean/std`，advantage 可能也变成 `nan`。
- `writer` 是全局变量，在 `train_step` 里直接使用。更稳的写法是把它作为 `GRPOTrainer` 成员。
- `CUDA_VISIBLE_DEVICES` 同时在文件顶部和 `__main__` 里设置，最好统一放到程序最开始，或改为命令行参数。
- 模型路径、数据路径都是硬编码，换机器运行前必须修改。
- `input_buffer` 只有在攒够 `gradient_accumulation_steps` 个 batch 后才训练；如果 dataloader 最后剩余 batch 不足，当前代码会跳过这部分剩余数据。
- `torch.cuda.empty_cache()` 每个 step 都调用可能影响性能，它只能释放缓存，不能释放仍被 tensor 引用的显存。

## 理解性问题与答案

**Q1：GRPO 为什么不需要 value model？**  
A：因为它对同一个 prompt 采样多条 response，然后用组内 reward 的均值和标准差构造相对 advantage。组内平均相当于 baseline，因此不用额外训练一个 value model 来估计状态价值。

**Q2：为什么同一个 prompt 要生成 `num_generations` 条回答？**  
A：GRPO 需要在同一个问题下比较多条回答的相对好坏。只有一条回答时无法计算组内相对优势。

**Q3：`attention_mask` 和 `action_mask` 的区别是什么？**  
A：`attention_mask` 给 Transformer 前向使用，覆盖 `prompt + response`，主要 mask pad；`action_mask` 给 loss 使用，只覆盖 response，mask eos/pad，让无效生成 token 不参与策略梯度。

**Q4：`old_action_log_probs` 为什么叫 old？**  
A：它是在采样 rollout 后记录的策略概率。后面训练时模型参数会变化，当前策略 log prob 会变成新的 `action_log_probs`，PPO 需要比较新旧策略概率比。

**Q5：`advantages.unsqueeze(1)` 是什么作用？**  
A：`advantages` 原本是 `[batch]`，每条 response 一个值；`coef_1` 是 `[batch, num_actions]`，每个 token 一个 ratio。`unsqueeze(1)` 把 advantage 变成 `[batch, 1]`，从而广播到每个 token。

**Q6：为什么 `get_action_log_probs` 要用 `input_ids[:, 1:]`？**  
A：因为 decoder-only LM 的 `logits[:, t, :]` 预测的是下一个 token，也就是 `input_ids[:, t + 1]`。因此真实 label 要整体右移一位。

**Q7：为什么最后取 `[:, -num_actions:]`？**  
A：完整输入是 `prompt + response`，response 在序列末尾；GRPO 只训练模型自己生成的 response token，所以只取最后 `num_actions` 个 token 的 log prob。

**Q8：为什么可以对同一批经验训练多轮？这不是离线了吗？**  
A：这是 PPO/GRPO 的短期 rollout 复用。先用当前模型采样一小批经验，再训练 `num_iterations` 轮，然后用更新后的模型重新采样。它不是长期固定数据集的离线 RL，但如果 `num_iterations` 过大，经验会变得 stale。

**Q9：`input_buffer` 的作用是什么？**  
A：它缓存 `gradient_accumulation_steps` 个 batch 的经验，用于梯度累积，也支持同一批 rollout 被 `num_iterations` 次复用。

**Q10：`clip_eps` 在 loss 中解决什么问题？**  
A：它限制 `ratio = π_new / π_old` 的范围，防止当前策略相对采样时旧策略变化过大，从而提高训练稳定性。

**Q11：`beta` 和 reference model 有什么关系？**  
A：`beta != 0` 时会复制一份 reference model，计算当前策略相对参考策略的 KL 惩罚。`beta` 越大，模型越不容易偏离初始模型。

**Q12：为什么 reward 是句子级，但 loss 是 token 级？**  
A：奖励函数通常只能对整条回答打分，但语言模型训练需要对每个生成 token 的概率做梯度更新。所以代码把同一条 response 的 advantage 广播给这条 response 的所有有效 token。

**Q13：`gather` 在 `get_action_log_probs` 里取的是什么？**  
A：模型每个位置都会输出整个词表的 log probability。`gather` 根据真实 token id，从词表维度取出“模型给真实生成 token 分配的 log probability”。

**Q14：为什么 `loss` 要除以 `action_mask.sum(dim=1)`？**  
A：不同 response 的有效长度可能不同。先按每条 response 的有效 token 平均，可以避免长回答因为 token 多而天然贡献更大的 loss。

**Q15：当前代码最应该先改的点是什么？**  
A：优先在 `generate` 中显式设置 `do_sample=True`，补齐 tokenizer 的 `pad_token` 处理，并保护 `action_mask.sum(dim=1) == 0` 和 reward `nan` 的情况。

## 运行前准备

当前脚本是学习版代码，路径和设备号写在文件里。运行前需要根据自己的机器修改：

- `train.py` 里的 `CUDA_VISIBLE_DEVICES`
- `AutoTokenizer.from_pretrained(...)` 和 `AutoModelForCausalLM.from_pretrained(...)` 的模型路径
- `GSM8KDataset(...)` 的数据集路径
- 如 tokenizer 没有 `pad_token`，建议显式设置为 `eos_token`

依赖方向：

```bash
pip install torch transformers datasets tensorboard
```

启动训练：

```bash
cd grpo_from_scratch
python train.py
```

训练日志会写入 `runs/`，模型会按 `save_steps` 保存到 `output/`。

## 学习顺序

建议按这个顺序阅读：

1. 先看 [GRPO.md](./GRPO.md)，理解 GRPO 为什么可以不用 value model。
2. 再看 `reward_func.py`，理解规则奖励如何提供训练信号。
3. 然后看 `train.py` 中的 `generate_samples`、`generate_experiences`、`get_action_log_probs`、`compute_loss`、`train_step`、`train`。
4. 最后对照本文的易错点检查实现细节。

## 相关笔记

[infer_py讲解.md](./infer_py讲解.md) 是对 Search-R1 推理脚本的拆解，重点解释模型如何在生成过程中触发 `<search>`、调用检索服务，并把 `<information>` 拼回上下文继续推理。它不是本仓库 GRPO trainer 的直接依赖，但可以帮助理解基于搜索增强推理的 RL 训练/推理范式。
