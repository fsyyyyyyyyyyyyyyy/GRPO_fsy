from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class GSM8KDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        
        self.tokenizer = tokenizer
        data = load_dataset(data_path)
        self.data = data['train']
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        # prompt = self.tokenizer.apply_chat_template(sample['prompt'], tokenize=False, add_generation_prompt=True)
        answer = sample['answer_only']
        prompt = sample['question_zh-cn']
        return {'prompt': prompt, 'answer': answer}

# Samples 这个类是用来组织和保存每一个训练样本结构的数据。
@dataclass   # 这是一个装饰器，它让我们不需要写__init__等方法，就能自动生成这些方法，方便地创建数据类。
class Samples:
    prompt_response_ids: torch.Tensor            # 这个保存了完整的prompt和生成response拼接后的token id序列，类型是PyTorch的Tensor张量。
    response_ids: torch.Tensor                   # 这个保存了仅有response部分的token id序列，也是PyTorch的Tensor。
    prompt: Any                                 # 这个变量存储原始的prompt（问题、输入文本），类型可以是任意类型（Any），通常是字符串。
    answer: Any                                 # 这个变量存储标准答案或ground truth，类型可以是任意类型（Any），通常是字符串。
    attention_mask: Optional[torch.LongTensor]   # 这个是可选的（Optional），如果提供了，就是一个标记哪些token应被模型关注的mask（掩码），类型是长整型张量（LongTensor）。
    action_mask: Optional[torch.BoolTensor]      # 这个也是可选的，是一个布尔类型的张量（BoolTensor），标记哪些位置是动作（用于策略梯度、PPO等）。
    num_actions: Union[int, torch.Tensor]        # 这个可以是一整个整数或者张量，描述该样本中包含的动作数量（如生成的token数量）。
    response_length: int                         # 这是单个response的长度，即生成序列的token数量，类型是整数。


class GRPOArguments:
    
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数
    max_prompt_length = 256 # 最大输入长度
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 2 # 梯度累加
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1

class GRPOTrainer:
    """
    最小版 GRPO 训练器。

    训练流程：
    1. 对每个 prompt 采样 num_generations 条回答，组成一个 group。
    2. 用奖励函数或奖励模型给每条回答打分。
    3. 在 group 内标准化奖励，得到 response 级别的 advantage。
    4. 用 PPO/GRPO 风格的 clipped objective 更新策略模型。
    """
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        # 保存训练参数，例如学习率、生成长度、KL 系数、梯度累积步数等。
        self.args = args
        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model) # 根据模型model或本地路径，加载一个因果语言模型
        # 策略模型，也就是训练过程中会被更新的语言模型。
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            # reference model 固定不训练，用来计算 KL 惩罚，限制策略模型偏离初始模型太远。
            self.ref_model = deepcopy(model)
            self.ref_model.eval() # 把参考模型切换到推理模式
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            # from_pretrained() 从一个已经训练好的模型/分词器名称，或者本地路径中，加载对应对象。
        
        # 统一 tokenizer 配置，当前只设置了 padding_side。
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        # 如果 reward_funcs 是字符串，则转换为列表
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs] # reward_funcs 是奖励模型路径或者 HuggingFace 模型名
        
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                # 字符串形式的 reward_func 被认为是奖励模型路径。
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device) # AutoModelForSequenceClassification 是加载一个序列分类模型
                # 奖励模型是分类/打分模型，输入一个文本，输出一个分数
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            # 普通 Python 奖励函数不需要 tokenizer，因此默认填 None。
            reward_tokenizers = [None] * len(reward_funcs)
            
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
            
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
            
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                # 奖励模型需要自己的 tokenizer；没有显式传入时，从模型配置路径加载。
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    # 有些 tokenizer 没有 pad token，这里复用 eos token 作为 pad。
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        # Adam 直接优化策略模型的全部参数。
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        # 缓存已经采样并计算好 reward/advantage/log_prob 的经验。
        # 当 num_iterations > 1 时，可以复用同一批经验训练多轮。
        self.input_buffer = [None] * self.args.gradient_accumulation_steps # 创建一个列表作为经验缓冲区
        
        # 模型更新的次数
        # 记录真正执行 optimizer.step() 的次数。
        self.update_steps = 0 
    def get_tokenizer(self, tokenizer):
        # decoder-only 模型批量生成时通常使用左 padding，使每条样本末尾对齐。
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位 rollout
    def generate_samples(self, inputs):
        """
        对一个 batch 内的每个 prompt 生成一组回答。

        返回 Samples 列表。列表中每个元素对应一个 prompt，
        元素内部包含 num_generations 条回答及其 mask。
        """
        samples_list = []
        self.model.eval() # 把模型切换到推理模式，用于生成回答
        # DataLoader 会把多个样本的 prompt 聚合成 inputs['prompt'] 列表。
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts) # 兼容某些数据集没有标准答案的情况
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        # 最终固定序列长度 = prompt 最大长度 + response 最大长度。
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        # 同时遍历问题和答案
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            # apply_chat_template 将 system/user 消息转换成模型期望的聊天格式文本。
            # tokenize=False 表示返回字符串；add_generation_prompt=True 会添加 assistant 起始标记。
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 生成一个group的输入数据
            # 同一个 prompt 复制 num_generations 份，一次性采样出一个 group。
            # return_tensors='pt' 表示返回 PyTorch Tensor。
            inputs = self.tokenizer(
                [input_text] * self.args.num_generations, # 同一个 prompt 复制 num_generations 份
                padding='max_length', 
                max_length=self.args.max_prompt_length, 
                truncation=True, 
                return_tensors='pt' # 返回 PyTorch tensor
            )
            prompt_ids = inputs['input_ids'] # 取出 prompt 的 token ids

            # 生成回答
            # 模型预测的概率分布里随机采样生成 token
            with torch.no_grad(): # 不计算梯度，生成样本时不需要反向传播，节省显存
                # generate 返回 prompt + response 的完整 token 序列。
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_generate_length,
                                    temperature=0.9, # 控制采样随机性
                                    top_p = 1,
                                    top_k = 50) # 每步只从概率最高的 50 个 token 中采样
                
            # 为了后续 torch.cat，所有生成序列都整理到同一个 max_length。
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length] # 生成序列的完整长度超过 max_length 就截断
            else:
                # 序列长度不够 max_length 就在后面补 pad token
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            # attention_mask 标记非 pad token。
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            # response_ids 只保留模型新生成的部分。
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            # action_mask 标记参与策略梯度计算的 response token，排除 eos 和 pad。
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        

            # 存储的是一个group的数据
            # 一个 Samples 对象保存同一个 prompt 的整组生成结果。
            samples = Samples(
                prompt_response_ids=prompt_response_ids, # 完整的token序列，包含 prompt + response, shape 为 [num_generations, max_prompt_length + max_generate_length]
                response_ids=response_ids, # 只包含response部分的token id序列，shape 为 [num_generations, max_generate_length]
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask, # 完整序列的mask，shape 为 [num_generations, max_prompt_length + max_generate_length]
                action_mask=action_mask, # 只包含response部分的mask，shape 为 [num_generations, max_generate_length]
                num_actions=action_mask.size(1), # response 序列长度，最多有多少个action token位置
                response_length=action_mask.float().sum(dim=-1) # response 序列长度，即有多少个参与策略梯度计算的token位置
            )
            samples_list.append(samples) 

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        """
        生成训练所需的经验数据。

        经验包括：
        - prompt_response_ids: prompt 和 response 拼接后的 token ids
        - old_action_log_probs: 采样时策略模型在 response token 上的 log_prob
        - ref_action_log_probs: reference model 的 log_prob，只有 beta != 0 时存在
        - advantages: group 内奖励标准化后的优势值
        """
        
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        
        # 这些列表先按 prompt 收集，最后在 batch 维度上拼接。
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        # 取出一个 prompt 的一组生成结果
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # 计算策略模型输出 token 的概率
                # 记录采样策略的 log_prob，后续 PPO ratio 会用到它。
                old_action_log_probs = self.get_action_log_probs(
                    self.model, 
                    prompt_response_ids, 
                    attention_mask, 
                    num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出token的概率
                    # 如果启用 KL 惩罚，也计算 reference model 的 log_prob。
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # 存储各个奖励函数在一个group内各个响应的奖励
                # rewards_per_func[i, j] 表示第 i 个奖励函数给第 j 条回答的分数。
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                # 将 response token 解码成文本，供奖励函数或奖励模型打分。
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts) # 把同一个prompt复制num_generations份,使它和多条response一一对应
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)] # 拼成完整文本 prompt+response, 给奖励模型打分时使用
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            # 奖励模型形式：先 tokenize prompt+response，再取分类模型 logits。
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        # 普通 Python 函数形式：直接接收文本 prompt、response 和标准答案。
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                # rewards_per_func: [num_funcs, num_generations]
                # 如果没有指定 reward_weights，则默认每个奖励函数权重为 1。
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # 乘以各个奖励函数的权重
                # 按权重合并多个奖励函数的输出。
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                
                # rewards: [num_funcs, num_generations]
                # 对所有奖励函数求和，得到每条 response 的最终奖励。
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO的优势是句子粒度的，而非token粒度的
                # GRPO 在 group 内标准化奖励：高于组均值则 advantage 为正。
                # 这里 advantage 是 response 级别，后续会广播到每个 token。
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0), # 模型输入
            "attention_mask": torch.cat(batch_attention_mask, dim=0), # Transformer attention mask
            "action_mask": torch.cat(batch_action_mask, dim=0), # 参与策略梯度计算的 response token 的 mask
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0), # 采样时策略模型在 response token 上的 log_prob
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None, # 参考模型在 response token 上的 log_prob
            "advantages": torch.cat(batch_advantages, dim=0), # 组内奖励标准化后的优势值
        }
    
    def compute_loss(self, model, inputs):
        """
        计算 GRPO/PPO 风格的策略损失。

        每条 response 的 advantage 会作用到该 response 的所有有效 token 上。
        当 beta != 0 时，额外加入相对 reference model 的 KL 惩罚项。
        """
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        # 当前策略模型在 response token 上的 log_prob。
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            # k3 是一种非负 KL 估计形式：exp(log_ratio) - 1 - log_ratio。
            k3 = log_ratio.exp() - 1 - log_ratio # 恒为非负
        
        advantages = inputs['advantages']
        
        # num_iterations > 1 时，old_action_log_probs 固定为采样时的旧策略概率。
        # 否则只训练一次，可以直接 detach 当前 log_prob。
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 重要性采样 shape: [batch_size * num_generations, num_actions]
        # clip 后的 PPO ratio，用来限制单次更新幅度。
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 一个序列中每个token的优势是一样的
        # 同一条 response 的所有 token 使用同一个 response-level advantage。
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # pad/eos 后面的 token 不参与 loss。
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        # 先对每条 response 的有效 token 求平均，再对 batch 求平均。
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # shape: [batch_size * num_generations]
        loss = loss.mean()
        
        # loss = per_token_loss.sum() / action_mask.sum()
        
        return loss

    # 给定一段完整序列 prompt + response，计算模型对 response 中每一个实际生成 token 的 log probability
    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        """
        取出模型对实际生成 token 的 log probability。

        input_ids 是 prompt + response 的完整序列。
        num_actions 是 response 部分长度，因此这里只返回最后 num_actions 个位置。
        """
        
        # 计算策略模型输出token的概率
        # logits[:, t, :] 表示模型在位置 t 预测下一个 token 的分布。
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        # 去掉最后一个位置，因为最后一个 logits 没有对应的下一个 label。
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        # input_ids[:, 1:] 是每个位置真实出现的下一个 token。
        # gather 从词表维度取出真实 token 对应的 log_prob。
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        # squeeze 后形状为 [batch, seq_len - 1]，最后截取 response token 的 log_prob。
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        """执行一次反向传播；满足梯度累积步数时更新模型参数。"""
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        # 梯度累积时需要把 loss 除以累积步数，保持梯度尺度稳定。
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            # 完成一次参数更新，并清空梯度。
            optimizer.step()
            optimizer.zero_grad()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
        
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        """完整训练循环：采样经验、缓存经验、按梯度累积策略更新模型。"""
        # 估算总更新步数，用于日志展示。
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for _ in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                
                # 先用当前模型采样回答并计算奖励、advantage、旧 log_prob。
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    # 可以对同一批采样经验重复训练 num_iterations 轮。
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        # 定期保存 checkpoint。
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del inputs
    def save_model(self):
        """保存最终策略模型和 tokenizer。"""
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    
    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    
    args = GRPOArguments()
    
    writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/user/Downloads/Qwen2.5-1.5B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    

    
    
    prompts_dataset = GSM8KDataset('/home/user/wyf/deepseek_learn/gsm8k_chinese', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    

