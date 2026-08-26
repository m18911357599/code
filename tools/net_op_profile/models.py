"""Representative common networks (untrained weights; structure matches production nets)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.models.detection as tvd


@dataclass
class Workload:
    name: str
    family: str
    description: str
    make_model: Callable[[], nn.Module]
    make_input: Callable[[], object]
    notes: str


class BertBlock(nn.Module):
    """Unfused BERT encoder block: explicit QKV / matmul / softmax (pre-fusion ATen)."""

    def __init__(self, d_model: int = 768, n_head: int = 12, d_ff: int = 3072):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.dk = d_model // n_head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q(x).view(b, s, self.h, self.dk).transpose(1, 2)
        k = self.k(x).view(b, s, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(b, s, self.h, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dk))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, s, d)
        x = self.ln1(x + self.o(ctx))
        x = self.ln2(x + self.ff2(F.gelu(self.ff1(x))))
        return x


class BertBase(nn.Module):
    def __init__(self, vocab: int = 30522, seq: int = 128, n_layer: int = 12):
        super().__init__()
        d = 768
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(512, d)
        self.typ = nn.Embedding(2, d)
        self.ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([BertBlock() for _ in range(n_layer)])
        self.pool = nn.Linear(d, d)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.ln(self.tok(input_ids) + self.pos(pos) + self.typ(token_type_ids))
        for blk in self.blocks:
            x = blk(x)
        return torch.tanh(self.pool(x[:, 0]))


class Gpt2Block(nn.Module):
    def __init__(self, d_model: int = 768, n_head: int = 12, d_ff: int = 3072):
        super().__init__()
        assert d_model % n_head == 0
        self.h = n_head
        self.dk = d_model // n_head
        self.ln1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        h, dk = self.h, self.dk
        xn = self.ln1(x)
        q = self.q(xn).view(b, s, h, dk).transpose(1, 2)
        k = self.k(xn).view(b, s, h, dk).transpose(1, 2)
        v = self.v(xn).view(b, s, h, dk).transpose(1, 2)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(dk))
        if causal:
            qlen, klen = q.size(-2), k.size(-2)
            mask = torch.ones(qlen, klen, device=x.device, dtype=torch.bool).triu(diagonal=1 + klen - qlen)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(b, s, d)
        x = x + self.o(ctx)
        x = x + self.ff2(F.gelu(self.ff1(self.ln2(x))))
        return x, k, v


class Gpt2Small(nn.Module):
    def __init__(self, vocab: int = 50257, n_layer: int = 12):
        super().__init__()
        d = 768
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(1024, d)
        self.blocks = nn.ModuleList([Gpt2Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.tok(input_ids) + self.pos(pos)
        for blk in self.blocks:
            x, _, _ = blk(x, causal=True)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1:])

    def forward_decode(
        self,
        input_ids: torch.Tensor,
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        b, s = input_ids.shape
        past = k_caches[0].size(2)
        pos = torch.arange(past, past + s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.tok(input_ids) + self.pos(pos)
        new_k, new_v = [], []
        for i, blk in enumerate(self.blocks):
            x, k, v = blk(x, k_cache=k_caches[i], v_cache=v_caches[i], causal=True)
            new_k.append(k)
            new_v.append(v)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1:])


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """4-level U-Net (medical / segmentation). Input 1x3x128x128."""

    def __init__(self, base: int = 32):
        super().__init__()
        self.e1 = nn.Sequential(ConvBNReLU(3, base), ConvBNReLU(base, base))
        self.e2 = nn.Sequential(ConvBNReLU(base, base * 2, s=2), ConvBNReLU(base * 2, base * 2))
        self.e3 = nn.Sequential(ConvBNReLU(base * 2, base * 4, s=2), ConvBNReLU(base * 4, base * 4))
        self.e4 = nn.Sequential(ConvBNReLU(base * 4, base * 8, s=2), ConvBNReLU(base * 8, base * 8))
        self.b = nn.Sequential(ConvBNReLU(base * 8, base * 16, s=2), ConvBNReLU(base * 16, base * 16))
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.d4 = nn.Sequential(ConvBNReLU(base * 16, base * 8), ConvBNReLU(base * 8, base * 8))
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = nn.Sequential(ConvBNReLU(base * 8, base * 4), ConvBNReLU(base * 4, base * 4))
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(ConvBNReLU(base * 4, base * 2), ConvBNReLU(base * 2, base * 2))
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(ConvBNReLU(base * 2, base), ConvBNReLU(base, base))
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.b(e4)
        d4 = self.d4(torch.cat([self.u4(b), e4], dim=1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


class Gpt2PrefillWrap(nn.Module):
    def __init__(self, inner: Gpt2Small):
        super().__init__()
        self.inner = inner

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.inner.forward_prefill(ids)


class Gpt2DecodeWrap(nn.Module):
    def __init__(self, inner: Gpt2Small, past: int = 127):
        super().__init__()
        self.inner = inner
        self.past = past
        h, dk = 12, 64
        for i in range(12):
            self.register_buffer(f"k{i}", torch.zeros(1, h, past, dk))
            self.register_buffer(f"v{i}", torch.zeros(1, h, past, dk))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        k_caches = [getattr(self, f"k{i}") for i in range(12)]
        v_caches = [getattr(self, f"v{i}") for i in range(12)]
        return self.inner.forward_decode(ids, k_caches, v_caches)


def _vision(ctor, size: int):
    def make_model():
        try:
            return ctor(weights=None)
        except TypeError:
            return ctor()

    def make_input():
        return torch.randn(1, 3, size, size)

    return make_model, make_input


def build_workloads() -> list[Workload]:
    r50_m, r50_i = _vision(tvm.resnet50, 224)
    mnv3_m, mnv3_i = _vision(tvm.mobilenet_v3_large, 224)
    vit_m, vit_i = _vision(tvm.vit_b_16, 224)

    def ssd_model():
        return tvd.ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

    def ssd_input():
        return [torch.rand(3, 320, 320)]

    gpt = Gpt2Small()

    return [
        Workload(
            "ResNet-50",
            "CNN",
            "ImageNet 分类；残差 CNN 代表。输入 1×3×224×224。",
            r50_m,
            r50_i,
            "53 个 Conv + 等量 BN；残差 Add；末尾 FC。",
        ),
        Workload(
            "MobileNetV3-Large",
            "CNN-DW",
            "深度可分 / SE 轻量 CNN。输入 1×3×224×224。",
            mnv3_m,
            mnv3_i,
            "depthwise conv + pointwise 1×1 + h-swish/SE。",
        ),
        Workload(
            "ViT-B/16",
            "ViT",
            "Vision Transformer Base，patch=16。输入 1×3×224×224。",
            vit_m,
            vit_i,
            "torchvision 实现，Attention 常走融合 SDPA。",
        ),
        Workload(
            "BERT-Base",
            "Encoder",
            "12×768/12-head/FFN=3072；seq=128，unfused QKV。",
            lambda: BertBase(),
            lambda: (
                torch.randint(0, 30522, (1, 128)),
                torch.zeros(1, 128, dtype=torch.long),
            ),
            "显式 matmul+softmax，对应 lowering 前 ATen 图。",
        ),
        Workload(
            "GPT-2-Small-prefill",
            "Decoder",
            "GPT-2 Small 预填充，seq=128，因果 Attention。",
            lambda: Gpt2PrefillWrap(gpt),
            lambda: torch.randint(0, 50257, (1, 128)),
            "整段并行；含因果 mask + LM head。",
        ),
        Workload(
            "GPT-2-Small-decode",
            "Decoder",
            "自回归 1 token，KV cache 长度 127。",
            lambda: Gpt2DecodeWrap(gpt, past=127),
            lambda: torch.randint(0, 50257, (1, 1)),
            "GEMV/瘦矩阵 + KV cat；带宽敏感。",
        ),
        Workload(
            "SSDLite320",
            "Det",
            "MobileNetV3-Large 骨干 + SSDLite 头，320×320。",
            ssd_model,
            ssd_input,
            "分类/回归头 + NMS（TopK/sort）。eval 模式。",
        ),
        Workload(
            "U-Net",
            "Seg",
            "4 级编码-解码分割网。输入 1×3×128×128。",
            lambda: UNet(base=32),
            lambda: torch.randn(1, 3, 128, 128),
            "ConvTranspose 上采样 + skip cat。",
        ),
        Workload(
            "LSTM-2L",
            "RNN",
            "2 层 LSTM，hidden=512，seq=128。",
            lambda: nn.LSTM(512, 512, num_layers=2, batch_first=True),
            lambda: torch.randn(1, 128, 512),
            "门控逐时间步；算子次数随 seq 放大。",
        ),
    ]
