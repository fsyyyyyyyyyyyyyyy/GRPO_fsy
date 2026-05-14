# GRPO_fsy

这个仓库整理了我学习和手写 GRPO（Group Relative Policy Optimization）的笔记与最小 PyTorch 实现。重点不是复现一个完整工业级 RLHF 框架，而是把 GRPO 的核心训练逻辑拆开：同一个 prompt 采样多条回答，在组内比较奖励高低，用相对优势更新模型。

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

更完整的推导、tensor shape、代码对应关系和易错点见 [GRPO.md](./GRPO.md)。

## 代码入口

主要实现位于 [grpo_from_scratch/train.py](./grpo_from_scratch/train.py)：

| 模块 | 作用 |
|---|---|
| `GSM8KDataset` | 读取中文 GSM8K 数据，返回 `prompt` 和 `answer` |
| `Samples` | 保存同一个 prompt 下的一组生成样本 |
| `GRPOArguments` | 配置学习率、生成条数、序列长度、KL 系数等超参 |
| `GRPOTrainer.generate_samples` | 对同一个 prompt 采样多条回答 |
| `GRPOTrainer.generate_experiences` | 计算 reward、advantage、old log prob |
| `GRPOTrainer.compute_loss` | 计算 GRPO/PPO 风格的 clipped policy loss |
| `GRPOTrainer.train` | 外层训练循环 |

奖励函数位于 [grpo_from_scratch/reward_func.py](./grpo_from_scratch/reward_func.py)：

| 奖励函数 | 作用 |
|---|---|
| `correctness_reward` | `<answer>` 中的答案与标准答案完全一致时给高分 |
| `digit_reward` | 输出答案是数字时给稠密奖励 |
| `hard_format_reward` | 完整匹配 `<think>...</think><answer>...</answer>` 格式 |
| `mark_reward` | 对关键标签逐项给分，缓解格式奖励过稀疏的问题 |

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
3. 然后看 `train.py` 中的 `generate_samples`、`generate_experiences`、`compute_loss`。
4. 最后对照 `GRPO.md` 的“最容易犯的 10 个错误”检查实现细节。

## 相关笔记

[infer_py讲解.md](./infer_py讲解.md) 是对 Search-R1 推理脚本的拆解，重点解释模型如何在生成过程中触发 `<search>`、调用检索服务，并把 `<information>` 拼回上下文继续推理。它不是本仓库 GRPO trainer 的直接依赖，但可以帮助理解基于搜索增强推理的 RL 训练/推理范式。

## 当前代码可以继续改进的点

- 在 `generate(...)` 中显式设置 `do_sample=True`。
- 为缺少 `pad_token` 的 tokenizer 自动补齐配置。
- 把模型路径、数据路径、设备号改成命令行参数。
- 给 search / reward / dataset 相关逻辑加更稳的异常处理。
- 将 `writer` 从全局变量改为 trainer 内部成员。
