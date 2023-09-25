import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

QWEN7B_MODEL_PATH = Path("./hf/Qwen-7B-Chat").expanduser()

def make_data_qwen7b_wte():
    from modeling_qwen import QWenModel
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(QWEN7B_MODEL_PATH, trust_remote_code=True)
    config.vocab_size = 48
    config.hidden_size = 256

    m = QWenModel(config).float().eval()
    m.wte.weight.data.normal_(mean=0.0, std=0.02)

    seq_len = 3
    x = torch.arange(seq_len, dtype=torch.int64)[None, :]
    with torch.no_grad():
        y = m.wte(x)

    with open(HERE / "data/qwen7b_wte.data", "wb") as f:
        m.wte.weight.data.numpy().tofile(f)
        x.int().numpy().tofile(f)
        y.numpy().tofile(f)

def make_data_qwen7b_attn():
    from modeling_qwen import QWenAttention, apply_rotary_pos_emb, RotaryEmbedding
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(QWEN7B_MODEL_PATH, trust_remote_code=True)
    config.hidden_size = 32
    config.intermediate_size = config.hidden_size * 3
    config.num_attention_heads = 8
    config.kv_channels = 4
    config.no_bias = True
    config.use_flash_attn = False
    config.use_dynamic_ntk = False
    config.use_logn_attn = False
    config.fp32 = True
    head_dim = config.hidden_size // config.num_attention_heads

    def _split_heads(tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(tensor, num_heads, attn_head_size):
        tensor = tensor.contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    rotary_emb = RotaryEmbedding(4)
    rotary_pos_emb = rotary_emb(3)
    for idx in range(len(rotary_pos_emb)):
        rotary_pos_emb[idx] = rotary_pos_emb[idx]

    m = QWenAttention(config).float().eval()
    for param in m.parameters():
        param.data.uniform_(-0.5, 0.5)

    bs = 1
    seq_len = 3
    x = torch.randn(bs, seq_len, config.hidden_size)
    registered_causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).view(1, 1, seq_len, seq_len)
    with torch.no_grad():
        qkv = m.c_attn(x)
        q, k, v = qkv.split(config.hidden_size, dim=2)
        q = _split_heads(q, config.num_attention_heads, head_dim)
        k = _split_heads(k, config.num_attention_heads, head_dim)
        v = _split_heads(v, config.num_attention_heads, head_dim)

        cur_len = q.shape[1]
        rotary_pos_emb = [i[:, -cur_len:, :, :] for i in rotary_pos_emb]
        rotary_pos_emb = (rotary_pos_emb,) * 2
        q_pos_emb, k_pos_emb = rotary_pos_emb

        q = apply_rotary_pos_emb(q, q_pos_emb)
        k = apply_rotary_pos_emb(k, k_pos_emb)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        attn_scores = attn_scores / torch.full([], v.size(-1) ** 0.5,)

        q_len, k_len = q.size(-2), k.size(-2)
        causal_mask = registered_causal_mask[:, :, k_len - q_len : k_len, :k_len]
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_scores.dtype)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2)
        attn_output = _merge_heads(attn_output, config.num_attention_heads, head_dim)
        attn_output = m.c_proj(attn_output)

    with open(HERE / "data/qwen7b_attn.data", "wb") as f:
        m.c_attn.weight.data.numpy().tofile(f)
        m.c_attn.bias.data.numpy().tofile(f)
        m.c_proj.weight.data.numpy().tofile(f)
        x.numpy().tofile(f)
        attn_output.numpy().tofile(f)

def make_data_qwen7b_mlp():
    from modeling_qwen import QWenMLP
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(QWEN7B_MODEL_PATH, trust_remote_code=True)
    config.hidden_size = 32
    config.intermediate_size = config.hidden_size * 3
    config.num_hidden_layers = 1
    config.fp32 = True

    m = QWenMLP(config).float().eval()
    m.w1.weight.data.normal_(mean=0.0, std=0.02)
    m.w2.weight.data.normal_(mean=0.0, std=0.02)
    m.c_proj.weight.data.normal_(mean=0.0, std=0.02)

    x = torch.randn(3, 32)
    with torch.no_grad():
        y = m(x)

    with open(HERE / "data/qwen7b_mlp.data", "wb") as f:
        m.w1.weight.data.numpy().tofile(f)
        m.w2.weight.data.numpy().tofile(f)
        m.c_proj.weight.data.numpy().tofile(f)
        x.numpy().tofile(f)
        y.numpy().tofile(f)

def make_data_qwen7b_block():
    from modeling_qwen import QWenModel
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(QWEN7B_MODEL_PATH, trust_remote_code=True)
    config.hidden_size = 32
    config.num_attention_heads = 8
    config.intermediate_size = config.hidden_size * 3
    config.num_hidden_layers = 1
    config.torch_dtype = torch.float32
    config.vocab_size = 5

    m = QWenModel(config).eval()

    for param in m.parameters():
        param.data.uniform_(-0.5, 0.5)

    seq_len = 3

    # self attention
    x1 = torch.arange(seq_len, dtype=torch.int64)[None, :]
    attn_mask = torch.ones(1, seq_len, dtype=torch.int64)
    with torch.no_grad():
        out = m(x1, attention_mask=attn_mask, use_cache=True)
        y1 = out.last_hidden_state
        kv_cache = out.past_key_values

    # cross attention
    x2 = torch.tensor([[seq_len]], dtype=torch.int64)
    attn_mask = torch.ones(1, seq_len + 1, dtype=torch.int64)
    with torch.no_grad():
        out = m(x2, attention_mask=attn_mask, past_key_values=kv_cache, use_cache=True)
        y2 = out.last_hidden_state
        kv_cache = out.past_key_values

    # cross attention
    x3 = torch.tensor([[seq_len + 1]], dtype=torch.int64)
    attn_mask = torch.ones(1, seq_len + 2, dtype=torch.int64)
    with torch.no_grad():
        out = m(x3, attention_mask=attn_mask, past_key_values=kv_cache, use_cache=True)
        y3 = out.last_hidden_state
        kv_cache = out.past_key_values

    print(m)

    with open(HERE / "data/qweb7b_block.data", "wb") as f:
        m.wte.weight.data.numpy().tofile(f)
        m.layers[0].ln_1.weight.data.numpy().tofile(f)
        m.layers[0].attn.c_attn.weight.data.numpy().tofile(f)
        m.layers[0].attn.c_attn.bias.data.numpy().tofile(f)
        m.layers[0].attn.c_proj.weight.data.numpy().tofile(f)
        m.layers[0].ln2.weight.data.numpy().tofile(f)
        m.layers[0].mlp.w1.weight.data.numpy().tofile(f)
        m.layers[0].mlp.w2.weight.data.numpy().tofile(f)
        m.layers[0].mlp.c_proj.weight.data.numpy().tofile(f)
        m.ln_f.weight.data.numpy().tofile(f)

        x1.int().numpy().tofile(f)
        y1.numpy().tofile(f)
        x2.int().numpy().tofile(f)
        y2.numpy().tofile(f)
        x3.int().numpy().tofile(f)
        y3.numpy().tofile(f)

def main():
    torch.manual_seed(0)
    (HERE / "data").mkdir(parents=True, exist_ok=True)
    sys.path.append(str(QWEN7B_MODEL_PATH))
    make_data_qwen7b_mlp()
    make_data_qwen7b_block()
    make_data_qwen7b_wte()
    make_data_qwen7b_attn()

if __name__ == "__main__":
    main()
