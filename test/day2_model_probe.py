import torch

from pathlib import Path
import sys

# 让脚本在 test 目录下直接运行时，也能找到项目根目录里的 model 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def build_model():
    """创建一个最小可观察的 MiniMind 模型。

    这里不加载真实权重，只创建模型结构本身。
    目的不是测试模型效果，而是观察：
    1. 配置能否正常构建
    2. 模型主干是否能顺利前向传播
    3. 张量形状如何在各阶段变化
    """
    cfg = MiniMindConfig(
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
    )
    model = MiniMindForCausalLM(cfg).eval()
    return cfg, model


def build_dummy_input(cfg, batch_size=1, seq_len=8):
    """构造一组随机 token id。

    这里使用随机整数模拟 tokenizer 输出的 token ids。
    对 Day 2 来说，我们只关心：
    - 输入形状对不对
    - 模型能不能吃进去
    - 输出形状怎么变化
    """
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    return input_ids


def inspect_main_flow(model, input_ids, inspect_blocks=2):
    """观察主干流程中的关键张量形状。

    整个观察路径是：
    input ids
      -> embedding
      -> 前若干层 block
      -> final norm
      -> lm head
      -> logits

    我们只观察前两层 block，避免输出过长。
    """
    base = model.model

    print("=== Day2 Model Probe ===")
    print("input_ids:")
    print(input_ids)
    print("input_ids shape:", tuple(input_ids.shape))

    # 第一步：token id 进入 embedding，变成 hidden states。
    hidden_states = base.embed_tokens(input_ids)
    print("embedding shape:", tuple(hidden_states.shape))

    # 第二步：为当前序列长度准备对应位置的 RoPE 参数。
    position_embeddings = (
        base.freqs_cos[: input_ids.shape[1]],
        base.freqs_sin[: input_ids.shape[1]],
    )

    # 第三步：只观察前若干层 block，确认 hidden states 如何在层间流动。
    for i, layer in enumerate(base.layers[:inspect_blocks]):
        hidden_states, _ = layer(hidden_states, position_embeddings)
        print(f"after block {i + 1} shape:", tuple(hidden_states.shape))

    # 第四步：主干最后还会做一次 norm。
    hidden_states = base.norm(hidden_states)
    print("final norm shape:", tuple(hidden_states.shape))

    # 第五步：通过 lm_head 映射回词表空间，得到 logits。
    logits = model.lm_head(hidden_states)
    print("logits shape:", tuple(logits.shape))

    return logits


def validate_shapes(cfg, input_ids, logits):
    """做几个最关键的结构断言。

    这些断言不是在测模型效果，而是在测你对结构的理解有没有跑偏。
    """

    assert input_ids.dim() == 2, "input_ids 应该是二维张量: [batch, seq]"
    assert logits.dim() == 3, "logits 应该是三维张量: [batch, seq, vocab]"
    assert logits.shape[0] == input_ids.shape[0], "logits 的 batch 维度应和输入一致"
    assert logits.shape[1] == input_ids.shape[1], "logits 的 seq 维度应和输入一致"
    assert logits.shape[2] == cfg.vocab_size, "logits 最后一维应等于 vocab_size"


def print_summary(cfg, input_ids, logits):
    """打印最终观察结论。

    这里的输出重点不是日志漂亮，而是把今天看到的现象转成一句清晰结论。
    """
    print("\n=== Summary ===")
    print(f"hidden_size = {cfg.hidden_size}")
    print(f"num_hidden_layers = {cfg.num_hidden_layers}")
    print(f"vocab_size = {cfg.vocab_size}")
    print(f"input batch = {input_ids.shape[0]}")
    print(f"input seq_len = {input_ids.shape[1]}")
    print(f"logits last dim == vocab_size: {logits.shape[-1] == cfg.vocab_size}")
    print("结论: hidden states 在层间保持 hidden_size 维度流动，最后才映射到词表空间。")


def main():
    """脚本入口。

    Day 2 的最小目标是：
    1. 模型能正常实例化
    2. 随机输入能完成一条主链路
    3. 关键形状满足预期
    4. 你能从输出里读出结构逻辑
    """

    cfg, model = build_model()
    input_ids = build_dummy_input(cfg)

    # 关闭梯度，避免无意义的显存和计算开销。
    with torch.no_grad():
        logits = inspect_main_flow(model, input_ids)

    validate_shapes(cfg, input_ids, logits)
    print_summary(cfg, input_ids, logits)


if __name__ == "__main__":
    main()
