from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EMBEDDING_DIMENSION = 768
HEAD_COUNT = 12
HEAD_SIZE = 64
LAYER_COUNT = 12
EOS_TOKEN_ID = 65535


class Rwkv7Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        receptance: Tensor,
        decay: Tensor,
        key: Tensor,
        value: Tensor,
        in_context_key: Tensor,
        in_context_value: Tensor,
    ) -> Tensor:
        del ctx
        batch_size, token_count, hidden_size = receptance.shape
        state = torch.zeros(
            (batch_size, 12, 64, 64),
            dtype=torch.float32,
            device=receptance.device,
        )
        outputs: List[Tensor] = []
        for token_index in range(token_count):
            rr = receptance[:, token_index].view(batch_size, 12, 64, 1)
            ww = decay[:, token_index].view(batch_size, 12, 1, 64)
            kk = key[:, token_index].view(batch_size, 12, 1, 64)
            vv = value[:, token_index].view(batch_size, 12, 64, 1)
            aa = in_context_key[:, token_index].view(batch_size, 12, 64, 1)
            bb = in_context_value[:, token_index].view(batch_size, 12, 1, 64)
            decay_factor = torch.exp(-0.6065306597 * ww)
            projection = torch.matmul(torch.matmul(state, aa), bb)
            state = state * decay_factor + projection + torch.matmul(vv, kk)
            outputs.append(torch.matmul(state, rr).view(batch_size, hidden_size))
        return torch.stack(outputs, dim=1)

    @staticmethod
    def symbolic(
        graph: object,
        receptance: object,
        decay: object,
        key: object,
        value: object,
        in_context_key: object,
        in_context_value: object,
    ) -> object:
        output = graph.op(
            "com.localfind::Rwkv7",
            receptance,
            decay,
            key,
            value,
            in_context_key,
            in_context_value,
        )
        return output.setType(receptance.type())


class RwkvLayer(nn.Module):
    def __init__(self, state: dict[str, Tensor], layer_id: int):
        super().__init__()
        prefix = f"rwkv.blocks.{layer_id}."
        attention = f"{prefix}att."
        feed_forward = f"{prefix}ffn."
        self.is_first = layer_id == 0

        def add(name: str, key: str) -> None:
            self.register_buffer(name, state[key].float().contiguous())

        add("ln1_weight", f"{prefix}ln1.weight")
        add("ln1_bias", f"{prefix}ln1.bias")
        add("ln2_weight", f"{prefix}ln2.weight")
        add("ln2_bias", f"{prefix}ln2.bias")
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            add(name, f"{attention}{name}")
        for name in ("w0", "w1", "w2", "a0", "a1", "a2", "g1", "g2"):
            add(name, f"{attention}{name}")
        for name, fallback in (("v0", "a0"), ("v1", "a1"), ("v2", "a2")):
            key = f"{attention}{name}"
            add(name, key if key in state else f"{attention}{fallback}")
        add("k_k", f"{attention}k_k")
        add("k_a", f"{attention}k_a")
        add("r_k", f"{attention}r_k")
        add("receptance_weight", f"{attention}receptance.weight")
        add("key_weight", f"{attention}key.weight")
        add("value_weight", f"{attention}value.weight")
        add("output_weight", f"{attention}output.weight")
        add("attention_norm_weight", f"{attention}ln_x.weight")
        add("attention_norm_bias", f"{attention}ln_x.bias")
        add("ffn_x_k", f"{feed_forward}x_k")
        add("ffn_key_weight", f"{feed_forward}key.weight")
        add("ffn_value_weight", f"{feed_forward}value.weight")

    def time_mix(self, x: Tensor, value_first: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, token_count, hidden_size = x.shape
        previous = torch.zeros((batch_size, 1, hidden_size), dtype=x.dtype, device=x.device)
        shift = torch.cat((previous, x[:, :-1, :]), dim=1) - x
        receptance_input = x + shift * self.x_r
        decay_input = x + shift * self.x_w
        key_input = x + shift * self.x_k
        value_input = x + shift * self.x_v
        in_context_input = x + shift * self.x_a
        gate_input = x + shift * self.x_g

        receptance = F.linear(receptance_input, self.receptance_weight)
        decay = torch.tanh(torch.matmul(decay_input, self.w1))
        decay = torch.matmul(decay, self.w2)
        key = F.linear(key_input, self.key_weight)
        value = F.linear(value_input, self.value_weight)
        in_context = torch.sigmoid(self.a0 + torch.matmul(torch.matmul(in_context_input, self.a1), self.a2))
        gate = torch.matmul(torch.sigmoid(torch.matmul(gate_input, self.g1)), self.g2)

        normalized_key = F.normalize(
            (key * self.k_k).view(batch_size, token_count, 12, 64),
            p=2.0,
            dim=-1,
        ).view(batch_size, token_count, hidden_size)
        key = key * (1.0 + (in_context - 1.0) * self.k_a)
        if self.is_first:
            value_first = value
        else:
            value_mix = torch.sigmoid(
                self.v0 + torch.matmul(torch.matmul(value_input, self.v1), self.v2)
            )
            value = value + (value_first - value) * value_mix

        decay = torch.sigmoid(self.w0 + decay)
        mixed = Rwkv7Function.apply(
            receptance,
            decay,
            key,
            value,
            -normalized_key,
            normalized_key * in_context,
        )
        mixed = F.group_norm(
            mixed.view(batch_size * token_count, hidden_size),
            12,
            self.attention_norm_weight,
            self.attention_norm_bias,
            64e-5,
        ).view(batch_size, token_count, hidden_size)
        residual = (
            (receptance * key * self.r_k.view(1, 1, 12, 64).view(1, 1, hidden_size))
            .view(batch_size, token_count, 12, 64)
            .sum(dim=-1, keepdim=True)
            * value.view(batch_size, token_count, 12, 64)
        ).view(batch_size, token_count, hidden_size)
        return F.linear((mixed + residual) * gate, self.output_weight), value_first

    def channel_mix(self, x: Tensor) -> Tensor:
        batch_size, _, hidden_size = x.shape
        previous = torch.zeros((batch_size, 1, hidden_size), dtype=x.dtype, device=x.device)
        shift = torch.cat((previous, x[:, :-1, :]), dim=1) - x
        key = x + shift * self.ffn_x_k
        key = torch.relu(F.linear(key, self.ffn_key_weight)).square()
        return F.linear(key, self.ffn_value_weight)

    def forward(self, x: Tensor, value_first: Tensor) -> Tuple[Tensor, Tensor]:
        normalized = F.layer_norm(x, [768], self.ln1_weight, self.ln1_bias)
        attention, value_first = self.time_mix(normalized, value_first)
        x = x + attention
        normalized = F.layer_norm(x, [768], self.ln2_weight, self.ln2_bias)
        return x + self.channel_mix(normalized), value_first


class EmbeddingRwkvTiny(nn.Module):
    def __init__(self, state: dict[str, Tensor]):
        super().__init__()
        embedding = F.layer_norm(
            state["rwkv.emb.weight"].float(),
            [EMBEDDING_DIMENSION],
            state["rwkv.blocks.0.ln0.weight"].float(),
            state["rwkv.blocks.0.ln0.bias"].float(),
        )
        self.embedding = nn.Embedding.from_pretrained(embedding.contiguous(), freeze=True)
        self.layers = nn.ModuleList([RwkvLayer(state, layer_id) for layer_id in range(LAYER_COUNT)])
        self.register_buffer("output_norm_weight", state["rwkv.ln_out.weight"].float().contiguous())
        self.register_buffer("output_norm_bias", state["rwkv.ln_out.bias"].float().contiguous())
        self.register_buffer("retr_fc1_weight", state["head.retr_head.fc1.weight"].float().contiguous())
        self.register_buffer("retr_fc2_weight", state["head.retr_head.fc2.weight"].float().contiguous())
        self.register_buffer("retr_norm_weight", state["head.retr_head.norm.weight"].float().contiguous())
        self.register_buffer("retr_norm_bias", state["head.retr_head.norm.bias"].float().contiguous())

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens)
        value_first = torch.zeros_like(x)
        for layer in self.layers:
            x, value_first = layer(x, value_first)
        x = F.layer_norm(x[:, -1, :], [768], self.output_norm_weight, self.output_norm_bias)
        projected = F.linear(torch.relu(F.linear(x, self.retr_fc1_weight)), self.retr_fc2_weight)
        return F.layer_norm(x + projected, [768], self.retr_norm_weight, self.retr_norm_bias)


def verify(wrapper: nn.Module, checkpoint: Path, repository: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rwkv_audit", Path(__file__).with_name("audit-embedding-rwkv.py")
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    namespace, _ = audit.load_reference(repository)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    audit.verify(namespace, state, wrapper)


def export_vocab(repository: Path, output: Path) -> None:
    sys.path.insert(0, str(repository / "package" / "src"))
    from rwkv_emb.tokenizer import RWKVTokenizer

    tokenizer = RWKVTokenizer()
    with output.open("wb") as file:
        file.write(b"RWKVTOK1")
        file.write(struct.pack("<I", 65536))
        for token_id in range(65536):
            token = b"" if token_id == 0 else tokenizer.idx2token.get(token_id, b"")
            file.write(struct.pack("<H", len(token)))
            file.write(token)
    print(f"Exported {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    wrapper = EmbeddingRwkvTiny(state).eval()
    if not args.skip_verify:
        verify(wrapper, args.checkpoint, args.repository)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.tensor([[0, EOS_TOKEN_ID]], dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (sample,),
        args.output,
        input_names=["tokens"],
        output_names=["embedding"],
        dynamic_axes={"tokens": {0: "batch", 1: "tokens"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        custom_opsets={"com.localfind": 1},
    )
    print(f"Exported {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)")
    export_vocab(args.repository, args.output.with_name("rwkv_vocab.bin"))


if __name__ == "__main__":
    main()
