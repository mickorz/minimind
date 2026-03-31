import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 本文件的职责：
# 1. 定义 MiniMind 的配置结构。
# 2. 定义单层 Transformer Block 的组成方式。
# 3. 定义从 token id 到 logits 的前向推理链路。
# 4. 定义自回归生成文本时使用的采样逻辑。


# MiniMindConfig 配置初始化流程
#
# MiniMindConfig 的初始化流程是这样的：
#
# MiniMindConfig
#    ├─> 读取主干规模参数
#    │     ├─> hidden size
#    │     ├─> num hidden layers
#    │     └─> vocab size
#    ├─> 读取注意力参数
#    │     ├─> num attention heads
#    │     ├─> num key value heads
#    │     └─> head dim
#    ├─> 读取位置编码参数
#    │     ├─> rope theta
#    │     └─> rope scaling
#    └─> 读取 MoE 参数
#          ├─> num experts
#          ├─> num experts per tok
#          └─> router aux loss coef
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        """初始化 MiniMind 的结构配置。

        参数说明：
            hidden_size:
                隐藏层维度，也叫模型主通道宽度。
                可以理解成每个 token 在模型内部被表示成多长的向量。
            num_hidden_layers:
                Transformer Block 的层数。
                层数越多，模型能逐层提炼的表示越深。
            use_moe:
                是否启用 MoE。
                MoE 是 Mixture of Experts 的缩写，中文常译为“专家混合”。
                它的思路是让不同 token 只激活部分专家网络，以更低激活成本获得更强表达能力。
            **kwargs:
                其余扩展配置，主要包括：
                - dropout:
                    丢弃率。Dropout 是一种正则化手段，用来降低过拟合风险。
                - vocab_size:
                    词表大小，决定 logits 最后一维长度。
                - bos_token_id:
                    Begin Of Sequence，序列开始 token 的 id。
                - eos_token_id:
                    End Of Sequence，序列结束 token 的 id。
                - flash_attn:
                    是否优先使用 Flash Attention。
                    Flash Attention 是一种融合注意力实现，通常更快更省显存。
                - num_attention_heads:
                    Query 头数量，也就是注意力头数。
                - num_key_value_heads:
                    Key 和 Value 的头数。
                    当它小于 Query 头数时，属于 GQA。
                - head_dim:
                    每个注意力头的维度。
                - hidden_act:
                    前馈层激活函数名称，例如 silu。
                - intermediate_size:
                    前馈层中间维度，也叫扩展维度。
                - max_position_embeddings:
                    位置编码支持的最大长度。
                - rms_norm_eps:
                    RMSNorm 的数值稳定项，避免除零或极小数不稳定。
                - rope_theta:
                    RoPE 的频率底数。
                    RoPE 是 Rotary Position Embedding，中文常译为“旋转位置编码”。
                - inference_rope_scaling:
                    推理时是否启用 RoPE 外推。
                - num_experts:
                    MoE 的专家数量。
                - num_experts_per_tok:
                    每个 token 最多激活几个专家。
                - moe_intermediate_size:
                    MoE 专家内部前馈层的中间维度。
                - norm_topk_prob:
                    是否对选中的 top k 专家权重再归一化。
                - router_aux_loss_coef:
                    路由辅助损失权重，用于鼓励专家负载更均衡。
        """
        super().__init__(**kwargs)

        # 模型主干规模配置。
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # 注意力结构配置。
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)

        # 前馈层配置。
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)

        # 位置编码与归一化配置。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)

        # 推理时可选启用 YaRN 风格的 RoPE 外推。
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # MoE 配置。只有 use_moe=True 时这些参数才真正参与计算。
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# RMSNorm 数据归一化流程
#
# RMSNorm 的计算流程是这样的：
#
# 输入向量
#    ├─> 计算平方均值
#    ├─> 加上 eps
#    ├─> 开平方并取倒数
#    ├─> 与输入相乘做缩放
#    └─> 乘以可学习权重
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化 RMSNorm。

        参数说明：
            dim:
                需要归一化的最后一维大小。
                一般对应 hidden size 或 head dim。
            eps:
                数值稳定项。
                在分母很小时避免出现数值爆炸或除零问题。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        """执行均方根归一化核心计算。

        参数说明：
            x:
                输入张量。
                最后一维会被当成一个向量整体做 RMS 归一化。
        """
        # 这里不减均值，只按均方根做归一化。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """执行 RMSNorm 前向传播。

        参数说明：
            x:
                输入张量，通常是 hidden states 或 q k 张量。
        """
        # 先转成 float 计算，避免低精度下的数值不稳定；
        # 再转回原 dtype，兼顾稳定性和显存占用。
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预先生成 RoPE 所需的 cos 和 sin 表。

    参数说明：
        dim:
            单个注意力头的维度。
            RoPE 会按这个维度生成旋转频率。
        end:
            预生成到的最大位置长度。
            一般等于模型可支持的最大序列长度。
        rope_base:
            RoPE 的频率底数。
            底数越大，位置频率分布越平缓。
        rope_scaling:
            RoPE 外推配置。
            这里主要用于 YaRN。
            YaRN 是一种长上下文推理时常用的 RoPE 扩展方法。
    """
    # 预先生成 RoPE 需要的 cos 和 sin 表，避免训练和推理时重复计算。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        # YaRN 外推的核心思想：
        # 对高频和低频的缩放速度做分段控制，让长文本推理更稳定。
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    # t 是位置索引，freqs 是每个维度使用的频率。
    # 两者做 outer 之后，就得到了每个位置在每个维度上的旋转角度。
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # RoPE 最终使用的是 cos 和 sin，两者都扩展成完整 head dim。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """把旋转位置编码作用到 q 和 k 上。

    参数说明：
        q:
            Query 张量。
            Query 是注意力里“我要去找什么”的查询向量。
        k:
            Key 张量。
            Key 是注意力里“我能提供什么线索”的键向量。
        cos:
            对应位置的余弦表。
        sin:
            对应位置的正弦表。
        unsqueeze_dim:
            为了和 q k 的维度对齐，需要把 cos sin 插入到哪个维度。
    """
    # RoPE 作用在 q 和 k 上，而不是 v 上。
    # 这样位置关系会直接进入注意力分数的计算过程。
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把较少的 KV 头复制到和 Query 头数量一致。

    参数说明：
        x:
            Key 或 Value 张量。
            形状通常为 batch seq kv heads head dim。
        n_rep:
            每个 KV 头需要重复的次数。
            当 Query 头数大于 KV 头数时，这个值会大于 1。
    """
    # 这是 GQA 的典型写法：
    # Q 头很多，K 和 V 头较少，通过复制 KV 头减少参数量和显存占用。
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

# Attention 单层注意力流程
#
# Attention 的计算流程是这样的：
#
# 输入隐藏状态
#    ├─> q k v 线性投影
#    ├─> q k 做 RMSNorm
#    ├─> q k 叠加 RoPE
#    ├─> 拼接 past kv
#    ├─> 计算注意力分数
#    ├─> softmax 归一化
#    ├─> 对 v 做加权汇总
#    ├─> 输出投影
#    └─> 返回输出和 cache

class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """初始化单层自注意力模块。

        参数说明：
            config:
                MiniMind 的配置对象。
                这里会从中读取头数、head dim、dropout、是否启用 flash attention 等参数。
        """
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim

        # q k v 投影层，把 hidden size 投影到多头空间。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # 对 q 和 k 在 head 维度上再做一次归一化，提升训练稳定性。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 若当前 PyTorch 版本支持 flash attention，优先走融合实现。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """执行单层注意力计算。

        参数说明：
            x:
                输入隐藏状态。
                形状通常为 batch seq hidden size。
            position_embeddings:
                位置编码对，通常是 cos 和 sin。
                它们会被用于给 q 和 k 施加 RoPE。
            past_key_value:
                历史 KV 缓存。
                cache 是“缓存”的意思，这里保存已经算过的历史 key 和 value，
                用于增量解码时避免重复计算。
            use_cache:
                是否返回新的 KV 缓存。
                推理阶段通常开启，训练阶段通常关闭。
            attention_mask:
                注意力掩码。
                常见用途是屏蔽 padding 区域，防止模型关注无效位置。
        """
        bsz, seq_len, _ = x.shape
        # 第一步：线性投影得到 q k v。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 第二步：把最后一维重新整理成多头格式。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # 第三步：对 q 和 k 做归一化，再施加 RoPE。
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        # 第四步：推理阶段若启用 cache，就把历史 kv 接到前面。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        # 第五步：把 q k v 变成 attention 计算需要的形状。
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # 第六步：优先用融合注意力；否则走显式实现，方便阅读和兼容。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # scores 形状为 batch heads q len k len。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果掩码：当前位置不能看到未来 token。
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 额外 attention mask 通常用于 padding 掩码。
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        # 第七步：拼回 hidden size，并做输出投影。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

# FeedForward 前馈加工流程
#
# FeedForward 的计算流程是这样的：
#
# 输入隐藏状态
#    ├─> gate proj
#    ├─> up proj
#    ├─> 激活函数
#    ├─> 两路结果逐元素相乘
#    └─> down proj 回到隐藏维度
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        """初始化普通前馈层。

        参数说明：
            config:
                模型配置对象。
            intermediate_size:
                前馈层中间维度。
                若为 None，则使用配置里的默认中间维度。
        """
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """执行前馈层前向传播。

        参数说明：
            x:
                输入隐藏状态。
                形状通常为 batch seq hidden size。
        """
        # 这里使用的是常见的门控前馈结构：
        # 激活后的 gate 分支与 up 分支逐元素相乘，再映射回 hidden size。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

# MOEFeedForward 专家路由流程
#
# MOEFeedForward 的计算流程是这样的：
#
# 输入隐藏状态
#    ├─> gate 计算专家分数
#    ├─> 选择 top k 专家
#    ├─> 归一化专家权重
#    ├─> 逐专家执行前馈
#    ├─> 聚合专家输出
#    └─> 计算路由辅助损失
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """初始化 MoE 前馈层。

        参数说明：
            config:
                模型配置对象。
                其中应包含专家数量、每个 token 激活的专家数等 MoE 参数。
        """
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """执行 MoE 前馈层前向传播。

        参数说明：
            x:
                输入隐藏状态。
                形状通常为 batch seq hidden size。
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # 先为每个 token 计算应该送往哪个专家。
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 保证未命中的 expert 在训练图里依然有梯度占位。
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            # 负载均衡辅助损失，避免所有 token 都只走少数专家。
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

# MiniMindBlock 单层计算流程
#
# MiniMindBlock 的计算流程是这样的：
#
# 输入隐藏状态
#    ├─> input layernorm
#    ├─> self attention
#    ├─> 残差相加
#    ├─> post attention layernorm
#    ├─> mlp 或 moe mlp
#    ├─> 再次残差相加
#    └─> 输出本层结果
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        """初始化单层 Transformer Block。

        参数说明：
            layer_id:
                当前层编号。
                主要用于调试、日志或后续扩展时区分不同层。
            config:
                模型配置对象。
        """
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        """执行单层 Block 前向传播。

        参数说明：
            hidden_states:
                输入隐藏状态。
            position_embeddings:
                当前序列对应的位置编码对 cos 和 sin。
            past_key_value:
                历史 KV 缓存。
            use_cache:
                是否返回新缓存。
            attention_mask:
                注意力掩码。
        """
        # 第一段残差：注意力输出加回原输入。
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # 第二段残差：前馈输出加回注意力后的结果。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

# MiniMindModel 主干前向流程
#
# MiniMindModel 的计算流程是这样的：
#
# input ids
#    ├─> token embedding
#    ├─> dropout
#    ├─> 准备位置编码
#    ├─> 逐层执行 block
#    ├─> final norm
#    ├─> 汇总 moe aux loss
#    └─> 返回隐藏状态和 cache
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """初始化模型主干。

        参数说明：
            config:
                模型配置对象。
                主干会根据它构建 embedding、若干层 block、最终 norm 和 RoPE 表。
        """
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # cos 和 sin 在模型初始化时就预先算好，推理时只按位置切片。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        """执行模型主干前向传播。

        参数说明：
            input_ids:
                输入 token id 张量。
                形状通常为 batch seq。
            attention_mask:
                注意力掩码，常用于屏蔽 padding。
            past_key_values:
                历史 KV 缓存。
                增量推理时用它来减少重复计算。
            use_cache:
                是否返回新的缓存。
            **kwargs:
                预留扩展参数。
                当前主干里基本不依赖这些参数，但向上层接口保持兼容。
        """
        batch_size, seq_length = input_ids.shape

        # Hugging Face 某些 cache 对象会带 layers 字段，这里统一转成原始列表模式。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 推理阶段若已有 cache，需要从历史长度之后的位置继续取 RoPE。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # token id 先进入 embedding，变成模型可计算的隐藏向量。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)

        # dense 模型这里 aux loss 为 0；MoE 模型则把各层的路由辅助损失加起来。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

# MiniMindForCausalLM 因果语言模型流程
#
# MiniMindForCausalLM 的计算流程是这样的：
#
# input ids
#    ├─> MiniMindModel 主干
#    ├─> lm head 投影到词表
#    ├─> 可选计算训练 loss
#    ├─> 返回 logits
#    └─> 推理时进入 generate
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    def __init__(self, config: MiniMindConfig = None):
        """初始化因果语言模型封装。

        参数说明：
            config:
                模型配置对象。
                若为 None，则使用默认 MiniMindConfig。
        """
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 词嵌入层与输出层共享权重，减少参数量，也让输入输出语义空间更一致。
        self.model.embed_tokens.weight = self.lm_head.weight
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        """执行因果语言模型前向传播。

        参数说明：
            input_ids:
                输入 token id。
            attention_mask:
                注意力掩码。
            past_key_values:
                历史 KV 缓存。
            use_cache:
                是否返回新的缓存。
            logits_to_keep:
                只保留最后若干位置的 logits。
                logits 是未做 softmax 的原始得分。
                当值为 0 时，表示保留全部位置。
            labels:
                训练标签。
                若传入，则会额外计算交叉熵损失。
            **kwargs:
                其余扩展参数，会透传给主干模型。
        """
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)

        # logits_to_keep 允许在某些场景下只保留最后若干位置的 logits。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            # 训练时做 next token prediction：
            # 用当前位置的 logits 去预测下一个位置的 label。
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # 参考讨论：
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        """执行自回归生成。

        参数说明：
            inputs:
                初始输入 token。
                通常等价于 input_ids。
            attention_mask:
                注意力掩码。
            max_new_tokens:
                最多新生成多少个 token。
            temperature:
                温度系数。
                温度越高，分布越平，生成更随机；
                温度越低，分布越尖锐，生成更保守。
            top_p:
                nucleus sampling 的阈值。
                nucleus sampling 常译为“核采样”，表示只从累计概率达到阈值的一小批候选里采样。
            top_k:
                只保留得分最高的前 k 个候选。
            eos_token_id:
                序列结束 token 的 id。
            streamer:
                流式输出对象。
                用于边生成边把 token 传出去。
            use_cache:
                是否启用 KV 缓存。
            num_return_sequences:
                同一输入要生成多少条结果。
            do_sample:
                是否采样。
                True 时按概率采样，False 时走贪心解码。
            repetition_penalty:
                重复惩罚系数。
                值越大，越不容易重复已经生成过的 token。
            **kwargs:
                其余扩展参数。
                例如可通过 return_kv 控制是否把缓存一并返回。
        """
        # 增量生成的核心思路：
        # 1. 每次只喂给模型新增的 token。
        # 2. 利用 past kv cache 复用历史计算结果。
        # 3. 对最后一个位置的 logits 做采样或贪心选择。
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None

            # 只取最后一个位置的 logits 来决定下一个 token。
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                # 压低已经出现过的 token 分数，减少重复输出。
                for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
            if top_k > 0: 
                # top k 只保留分数最高的 k 个候选。
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                # top p 只保留累计概率不超过阈值的候选集合。
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None:
                # 已经结束的序列强制继续补 eos，便于批量生成时统一停止逻辑。
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
