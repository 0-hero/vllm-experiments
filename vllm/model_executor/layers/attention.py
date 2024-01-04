"""Multi-head attention."""
from typing import List, Optional

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip

import math
import torch.nn.functional as F
from torch import Tensor
from zeta.nn import YarnEmbedding

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
            )

        if input_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                # project the key and value tensors to the desired number of
                # heads.
                # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                key = key[:, :,
                          None, :].expand(key.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])

            # Set attention bias if not provided. This typically happens at the
            # very attention layer of every iteration.
            # FIXME(woosuk): This is a hack.
            if input_metadata.attn_bias is None:
                if self.alibi_slopes is None:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        [seq_len] * batch_size)
                    if self.sliding_window is not None:
                        attn_bias = attn_bias.make_local_attention(
                            self.sliding_window)
                    input_metadata.attn_bias = attn_bias
                else:
                    input_metadata.attn_bias = _make_alibi_bias(
                        self.alibi_slopes, self.num_kv_heads, batch_size,
                        seq_len, query.dtype)

            # TODO(woosuk): Too many view operations. Let's try to reduce them
            # in the future for code readability.
            if self.alibi_slopes is None:
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
            else:
                query = query.unflatten(0, (batch_size, seq_len))
                key = key.unflatten(0, (batch_size, seq_len))
                value = value.unflatten(0, (batch_size, seq_len))

            out = xops.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=input_metadata.attn_bias,
                p=0.0,
                scale=self.scale,
                op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
                (is_hip()) else None,
            )
            output = out.view_as(query)
        else:
            # Decoding run.
            if key_cache is not None and value_cache is not None:
                output = _paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                )
            else:
                # This happens during the initial memory profiling run for
                # CUDA graphs.
                output = torch.zeros_like(query)

        # Reshape the output tensor.
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype, device="cuda")
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> torch.Tensor:
    output = torch.empty_like(query)

    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
        (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
        _PARTITION_SIZE)
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    use_v1 = input_metadata.max_context_len <= 8192 and (
        max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    return output

def causal_mask(size):
    return torch.tril(torch.ones(size, size)).type(torch.bool)


def positional_encoding(seq_len, dim):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def apply_pos_emcode(tensor, pos, dim):
    seq_len = tensor.size(1)
    pe = positional_encoding(seq_len, dim)
    pe = pe.unsqueeze(0)
    return tensor + pe[:, pos]


class SelfExtendAttn(nn.Module):
    """
    SelfExtendAttn module performs self-attention on input sequences using
    both normal self-attention and grouped self-attention.

    Args:
        dim (int): The dimension of the input sequences.
        g_size (int): The size of each group for grouped self-attention.
        w_size (int): The window size for grouped self-attention.
        qk_norm (bool, optional): Whether to apply layer normalization to
            query and key vectors. Defaults to False.

    Example:
    import torch
    from se_attn import SelfExtendAttn

    # Example usage
    dim = 512  # Dimension of model
    g_size = 2  # Group size
    w_size = 4  # Window size for neighbor tokens
    self_extend = SelfExtendAttn(dim, g_size, w_size, qk_norm=True)

    # Example tensors for q, k, v, and pos
    q = torch.randn(1, 10, dim)
    k = torch.randn(1, 10, dim)
    v = torch.randn(1, 10, dim)
    pos = torch.arange(0, 10).unsqueeze(0)  # Example positional indices

    output = self_extend(q, k, v, pos)
    print(output)

    """

    def __init__(
        self,
        dim: int,
        g_size: int,
        w_size: int,
        qk_norm: bool = False,
        yarn_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        super(SelfExtendAttn, self).__init__()
        self.dim = dim
        self.g_size = g_size
        self.w_size = w_size
        self.qk_norm = qk_norm
        self.yarn_embeddings = yarn_embeddings

        # Layer normalization
        if qk_norm:
            self.norm = nn.LayerNorm(dim)

        # Yarn embeddings
        if yarn_embeddings:
            self.yarn = YarnEmbedding(dim, *args, **kwargs)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, pos: Tensor
    ) -> Tensor:
        """
        Forward pass of the SelfExtendAttn module.

        Args:
            q (torch.Tensor): The query tensor of shape (batch_size, seq_len, dim).
            k (torch.Tensor): The key tensor of shape (batch_size, seq_len, dim).
            v (torch.Tensor): The value tensor of shape (batch_size, seq_len, dim).
            pos (torch.Tensor): The position tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, dim).
        """
        seq_len = q.size(1)

        # Apply layer normalization to query and key vectors
        if self.qk_norm:
            q = self.norm(q)
            k = self.norm(k)

        # Normal self-attention for neighbor tokens
        ngb_q = apply_pos_emcode(q, pos, self.dim)
        ngb_k = apply_pos_emcode(k, pos, self.dim)
        ngb_attn = torch.matmul(ngb_q, ngb_k.transpose(-2, -1))
        ngb_attn = ngb_attn.masked_fill(
            ~causal_mask(seq_len), float("-inf")
        )

        # Grouped self-attention
        g_pos = pos // self.g_size
        shift = self.w_size - self.w_size // self.g_size
        s_g_pos = g_pos + shift
        g_q = apply_pos_emcode(q, s_g_pos, self.dim)
        g_k = apply_pos_emcode(k, g_pos, self.dim)
        g_attn = torch.matmul(g_q, g_k.transpose(-2, -1))
        g_attn = g_attn.masked_fill(
            ~causal_mask(seq_len), float("-inf")
        )

        # Create masks for merging attentions
        g_mask = torch.tril(
            torch.ones(seq_len - self.w_size, seq_len - self.w_size)
        )
        mask = torch.ones(seq_len, seq_len)
        mask[self.w_size :, : -self.w_size] -= g_mask

        # Merge attentions
        attn = torch.where(mask.bool(), ngb_attn, g_attn)
        attn_weights = F.softmax(attn, dim=-1)

        # Output
        output = torch.matmul(attn_weights, v)
        return output