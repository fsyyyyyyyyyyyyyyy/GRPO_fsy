# GRPO 代码实现笔记

## 1. `GSM8KDataset` 的作用

这段代码的核心作用：

```text
把 HuggingFace Dataset 包装成 PyTorch DataLoader 能读取的格式。
```

在 GRPO 里，数据集只需要提供两样东西：

| 字段 | 含义 |
|---|---|
| `prompt` | 给模型生成回答的问题 |
| `answer` | 标准答案，用来计算 reward |

代码：

```python
class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data["train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        answer = sample["answer_only"]
        prompt = sample["question_zh-cn"]
        return {"prompt": prompt, "answer": answer}
```

---

## 2. Dataset 三件套

PyTorch 自定义数据集一般固定写这三个部分：

| 方法 | 作用 |
|---|---|
| `__init__` | 加载数据、保存配置 |
| `__len__` | 返回数据集长度 |
| `__getitem__` | 根据索引返回一条样本 |

### `__init__`

```python
data = load_dataset(data_path)
self.data = data["train"]
```

含义：

```text
从 data_path 加载数据集，只取 train 训练集。
```

### `__len__`

```python
return len(self.data)
```

告诉 `DataLoader` 一共有多少条数据。

### `__getitem__`

```python
sample = self.data[index]
prompt = sample["question_zh-cn"]
answer = sample["answer_only"]
return {"prompt": prompt, "answer": answer}
```

每次根据 `index` 取一条样本，整理成训练代码需要的格式。

---

## 3. 为什么 GRPO Dataset 不提前 tokenize

SFT 里经常会在 Dataset 里提前 tokenize：

```text
text -> input_ids / attention_mask / labels
```

但 GRPO 不一样。

GRPO 的训练流程是：

```text
prompt 文本
 -> model.generate 生成多条 response
 -> reward 函数根据 response 和 answer 打分
 -> 组内标准化 reward 得到 advantage
 -> 计算 policy loss
```

所以 Dataset 最好返回原始文本：

```python
{
    "prompt": prompt,
    "answer": answer,
}
```

真正的 tokenize 放在 `generate_samples()` 里做，因为那里需要：

```text
同一个 prompt 复制 num_generations 份，再批量生成多个回答。
```

---

## 4. DataLoader 后的数据长什么样

单条样本：

```python
{
    "prompt": "小明有 3 个苹果，又买了 2 个，一共有几个？",
    "answer": "5",
}
```

如果：

```python
dataloader = DataLoader(dataset, batch_size=2)
```

那么 batch 会自动变成：

```python
{
    "prompt": ["问题1", "问题2"],
    "answer": ["答案1", "答案2"],
}
```

所以训练代码里可以写：

```python
prompts = [prompt for prompt in inputs["prompt"]]
answers = [answer for answer in inputs["answer"]]
```

---

## 5. 通用 Dataset 写法

更通用的 QA 数据集模板：

```python
from torch.utils.data import Dataset
from datasets import load_dataset


class QADataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer=None,
        split="train",
        prompt_key="question",
        answer_key="answer",
        use_chat_template=False,
    ):
        self.tokenizer = tokenizer
        self.data = load_dataset(data_path)[split]
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.use_chat_template = use_chat_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        prompt = sample[self.prompt_key]
        answer = sample[self.answer_key]

        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return {
            "prompt": prompt,
            "answer": answer,
        }
```

使用方式：

```python
dataset = QADataset(
    data_path="/path/to/dataset",
    tokenizer=tokenizer,
    split="train",
    prompt_key="question_zh-cn",
    answer_key="answer_only",
    use_chat_template=False,
)
```

---

## 6. 什么时候用 chat template

如果模型是 chat/instruct 模型，通常需要聊天模板：

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
```

但在当前 GRPO 代码里，chat template 放在 `generate_samples()` 中：

```python
input_text = self.tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    add_generation_prompt=True,
    tokenize=False,
)
```

因此 Dataset 里暂时不需要处理模板，只返回干净的问题文本即可。

---

## 7. 重点记忆

```text
Dataset 负责取数据，不负责训练逻辑。
```

对 GRPO 来说，Dataset 最重要的是返回：

```python
{"prompt": prompt, "answer": answer}
```

其中：

- `prompt` 用来让模型生成回答。
- `answer` 用来让 reward 函数判断回答是否正确。

最小通用结构：

```python
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = load_dataset(data_path)["train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return {
            "prompt": sample["question"],
            "answer": sample["answer"],
        }
```

