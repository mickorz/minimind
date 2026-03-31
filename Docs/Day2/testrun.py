from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
import torch

def inspect_main_flow(model, input_ids, inspect_blocks=2):


    base = model.model

    # [1] 输入形状
    print("input_ids shape:", tuple(input_ids.shape))

    # [2] token id 进入 embedding，变成 hidden states
    #     形状变化: [batch, seq] -> [batch, seq, hidden_size]
    hidden_states = base.embed_tokens(input_ids)
    print("embedding shape:", tuple(hidden_states.shape))

    # [3] 为当前序列长度准备对应位置的 RoPE 参数
    #     注意: RoPE 的 cos 和 sin 是预先算好的，这里按位置切片
    position_embeddings = (
        base.freqs_cos[: input_ids.shape[1]],
        base.freqs_sin[: input_ids.shape[1]],
    )

    # [4] 只观察前若干层 block，确认 hidden states 如何在层间流动
    #     每层 block 内部: layernorm -> attention -> 残差 -> layernorm -> mlp -> 残差
    #     但对主链来说，每层输出的形状始终是 [batch, seq, hidden_size]
    for i, layer in enumerate(base.layers[:inspect_blocks]):
        hidden_states, _ = layer(hidden_states, position_embeddings)
        print(f"after block {i + 1} shape:", tuple(hidden_states.shape))

    # [5] 主干最后还会做一次 RMSNorm
    #     形状不变，仍然是 [batch, seq, hidden_size]
    hidden_states = base.norm(hidden_states)
    print("final norm shape:", tuple(hidden_states.shape))

    # [6] 通过 lm_head 映射回词表空间，得到 logits
    #     形状变化: [batch, seq, hidden_size] -> [batch, seq, vocab_size]
    logits = model.lm_head(hidden_states)
    print("logits shape:", tuple(logits.shape))

    return logits




cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
model = MiniMindForCausalLM(cfg).eval()

input_ids = torch.randint(0, cfg.vocab_size, (1, 8))