import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class STSFullSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 8,
            n_heads: int = 1,
            kdim: int = None,
            vdim: int = None,
            dropout: float = 0.1,
            bias: bool = True,
            self_attention: bool = False,
    ):
        """
        Spatio-Temporal Synchronous Multi-Head Full Self-Attention Module
        [O(N^2) time-complexity]

        d_model: The dimension of Query, Key, Value for self-attention
        n_heads: The number of heads for calculating multi-head self attention
        """

        super(STSFullSelfAttention, self).__init__()

        self.d_model = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self.qkv_same_dim = self.kdim == d_model and self.vdim == d_model

        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // n_heads

        assert (
                self.head_dim * n_heads == self.d_model
        ), "Embedding size (d_model) needs to be divisible by n_heads"

        # n_heads (h): number of multi-attention heads
        # head_dim (d_k): dimension of projected q,k,v
        # d_model (C): dimension of the input q,k,v (dim of input X)

        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        # Define the projection matrix to generate Q, K, V
        self.W_K = nn.Linear(self.kdim, d_model, bias=bias)
        self.W_V = nn.Linear(self.vdim, d_model, bias=bias)
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)

        self.fc_out = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.W_K.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.W_V.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.W_Q.weight, gain=1/math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.W_K.weight)
            nn.init.xavier_uniform_(self.W_V.weight)
            nn.init.xavier_uniform_(self.W_Q.weight)

        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.0)

    def _q_k_dot_product(self, q, k, mask=None):
        """
        q: Query Matrix [B(Batch-Size) * h(n_heads), 12*N(Spatio-Temporal), d_k]
        k: Key Matrix [B(Batch-Size) * h(n_heads), 12*N(Spatio-Temporal), d_k]
        return: attn_probs: [B(Batch-Size) * h(n_heads), 12*N(Spatio-Temporal), 12*N]
        """

        bh, k_N, d_k = q.shape

        # [B*h, 12*N, d_k] -> [B*h, 12*N, 12*N]
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bh, k_N, k_N]

        '''if mask is not None:
            scores = scores * mask'''
        # print("Mask Shape: ", mask.shape)
        # print("Scores Shape: ", scores.shape)

        # [B*h, 12*N, 12*N]
        attn_weights_float = nn.Softmax(dim=-1)(attn_weights)
        # print("Attn weights Shape: ", attn_weights_float.shape)
        # raise ValueError("Attention weights ...")

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout(attn_weights)

        return attn_probs

    def _update_context(self, attn, v):
        """
        attn: Self-Attention Matrix [B(Batch-Size) * h(n_heads), 12*N(Spatio-Temporal), 12*N]
        v: Value Matrix [B(Batch-Size) * h(n_heads), 12*N(Spatio-Temporal), d_k]
        return: context [B(Batch-Size) * h(n_heads), 12*N(Spatio-Temporal), d_k]
        """

        bh, k_N, d_k = v.shape

        assert v is not None

        context = torch.bmm(attn, v)  # [B*h, 12*N, d_k]
        assert list(context.size()) == [bh, k_N, self.head_dim]

        return context

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor],
            value: Optional[torch.Tensor],
            mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        key: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        value: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        return context: [B(Batch-Size), 12*N(Spatio-Temporal), C(d_model)]
        """

        # input_q/k/v: [12*N, B, d_model (C)]
        k_N, B, d_model = query.shape

        assert d_model == self.d_model, f"query dim {d_model} != {self.d_model}"
        assert list(query.size()) == [k_N, B, d_model]

        # [12*N, B, d_model]
        q = self.W_Q(query)
        k = self.W_K(key)
        v = self.W_V(value)

        q *= self.scaling

        # [12*N, B, d_model] --> [12*N, B * h, d_k] --> [B * h, 12*N, d_k]
        q = (
            q.contiguous()
            .view(k_N, B * self.n_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(-1, B * self.n_heads, self.head_dim)
                .transpose(0, 1)
            )

        if v is not None:
            v = (
                v.contiguous()
                .view(-1, B * self.n_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == k_N

        # print("Value size: ", v.shape)
        # raise ValueError("Enter Correct Value ...")

        attn = self._q_k_dot_product(q, k, mask)  # [B*h, 12*N, 12*N]

        context = self._update_context(attn, v)  # [B*h, 12*N, d_k]

        context = context.transpose(0, 1).contiguous().view(k_N, B, d_model)
        out = self.fc_out(context).transpose(0, 1)  # [12*N, B, d_model (C)] -> [B, 12*N, d_model (C)]

        # print("out Shape: ", out.shape)
        # raise ValueError("output shape ...")

        return out


class STSProbSparseSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int = 8,
            n_heads: int = 1,
            kdim: int = None,
            vdim: int = None,
            factor: int = 5,
            dropout: float = 0.1,
            bias: bool = True,
            output_attention: bool = False,
            self_attention: bool = False,
    ):
        """
        Spatio-Temporal Synchronous Multi-Head Prob-Sparse Self-Attention Module
        [O(N*log(N)) time-complexity]

        d_model: The dimension of Query, Key, Value for self-attention
        n_heads: The number of heads for calculating multi-head self
        factor: The constant sampling factor c to control O(ln N) attentions
        attention_dropout: The dropout for the prob-sparse attention
        output_attention: Whether to return the output attention or not
        """

        super(STSProbSparseSelfAttention, self).__init__()

        self.d_model = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self.qkv_same_dim = self.kdim == d_model and self.vdim == d_model

        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // n_heads

        assert (
                self.head_dim * n_heads == self.d_model
        ), "Embedding size (d_model) needs to be divisible by n_heads"

        # n_heads (h): number of multi-attention heads
        # head_dim (d_k): dimension of projected q,k,v
        # d_model (C): dimension of the input q,k,v (dim of input X)

        self.scaling = self.head_dim ** -0.5

        self.output_attention = output_attention

        self.self_attention = self_attention

        assert self.self_attention, "Only support self attention"

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.factor = factor

        # Define the projection matrix to generate Q, K, V
        self.W_K = nn.Linear(self.kdim, d_model, bias=bias)
        self.W_V = nn.Linear(self.vdim, d_model, bias=bias)
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)

        self.fc_out = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.W_K.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.W_V.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.W_Q.weight, gain=1/math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.W_K.weight)
            nn.init.xavier_uniform_(self.W_V.weight)
            nn.init.xavier_uniform_(self.W_Q.weight)

        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.0)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(k_N)

        # Q [B*H, L (12*N), E (d_k)]
        # sample_k = factor*ln(k_N)
        # n_top = factor*ln(k_N)
        bh, k_N, E = K.shape

        B = bh // self.n_heads
        H = self.n_heads

        Q = (
            Q.contiguous()
            .view(B, H, k_N, E)
        )

        K = (
            K.contiguous()
            .view(B, H, k_N, E)
        )

        assert list(Q.size()) == [B, H, k_N, self.head_dim]
        assert list(K.size()) == [B, H, k_N, self.head_dim]

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, k_N, k_N, E)  # [B, H, k_N, k_N, self.head_dim]
        # print("K_expand shape: ", K_expand.shape)
        index_sample = torch.randint(k_N, (k_N, sample_k))  # [k_N, sample_k]
        # print("index_sample shape: ", index_sample.shape)

        K_sample = K_expand[:, :, torch.arange(k_N).unsqueeze(1), index_sample, :]
        # [B, H, k_N, sample_k, self.head_dim]
        # print("K_sample shape: ", K_sample.shape)

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # [B, H, k_N, sample_k]
        # print("Q_K_sample shape: ", Q_K_sample.shape)

        # find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), k_N)
        # [B, H, k_N]
        # print("Sparsity Measurement M shape: ", M.shape)

        M_top = M.topk(n_top, sorted=False)[1]  # select n_top indices from M -> M_top are the top indices
        # [B, H, n_top]
        # print("Top Sparsity Measurement M-Top shape: ", M_top.shape)

        # use the reduced Q to calculate Q_K: Q shape: [B, H, k_N, self.head_dim]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(k_N)
        # Q_reduce shape: [B, H, n_top, self.head_dim]
        # print("Q_reduce shape: ", Q_reduce.shape)

        # K shape: [B, H, k_N, self.head_dim]
        scores = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(k_N)*k_N
        # scores shape: [B, H, n_top, k_N]
        # print("scores shape: ", scores.shape)

        return scores, M_top

    def _get_initial_context(self, V, k_N):

        # V [B*H, k_N (12*N), D (d_k)]
        bh, k_N, D = V.shape

        B = bh // self.n_heads
        H = self.n_heads

        V = (
            V.contiguous()
            .view(B, H, k_N, D)
        )

        V_sum = V.mean(dim=-2)

        contex = V_sum.unsqueeze(-2).expand(B, H, k_N, V_sum.shape[-1]).clone()

        return contex

    def _update_context(self, context_in, V, scores, index, k_N):

        # V [B*H, k_N (12*N), D (d_k)]
        bh, k_N, D = V.shape

        B = bh // self.n_heads
        H = self.n_heads

        V = (
            V.contiguous()
            .view(B, H, k_N, D)
        )

        # print("Context shape: ", context_in.shape)
        # print("V shape: ", V.shape)
        # print("Scores Top : ", scores.shape)
        # print(scores[0, 0, :5, :5])
        # print("Index: ", index.shape)
        # print("Top Indices: ", index[0, 0, :5])
        # idx = index[0, 0, :5]

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        # print("Attns: ", attn.shape)
        # print(attn[0, 0, :5, :5])

        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)

        # print("Context_in after shape: ", context_in.shape)

        if self.output_attention:
            attns = (torch.ones([B, H, k_N, k_N]) / k_N).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            # print("Attns shape: ", attns.shape)
            # print(attns[0, 0, idx, :])
            # raise ValueError
            return context_in, attns
        else:
            return context_in, None

    def forward(
            self,
            query: torch.Tensor,
            key: Optional[torch.Tensor],
            value: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        query: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        key: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        value: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        return out: [B(Batch-Size), 12*N(Spatio-Temporal), C(d_model)]
        """

        # queries/keys/values: [12*N, B, d_model (C)]
        k_N, B, d_model = query.shape

        assert d_model == self.d_model, f"query dim {d_model} != {self.d_model}"
        assert list(query.size()) == [k_N, B, d_model]

        # [12*N, B, d_model]
        q = self.W_Q(query)
        k = self.W_K(key)
        v = self.W_V(value)

        q *= self.scaling

        # n_heads (h): number of multi-attention heads
        # head_dim (d_k): dimension of projected q,k,v
        # d_model (C): dimension of the input q,k,v (dim of input X)
        # head_dim * n_heads == d_model

        # [12*N, B, d_model] --> [12*N, B*h, d_k] --> [B*h, 12*N, d_k]
        q = (
            q.contiguous()
            .view(k_N, B * self.n_heads, self.head_dim)
            .transpose(0, 1)
        )

        if k is not None:
            k = (
                k.contiguous()
                .view(-1, B * self.n_heads, self.head_dim)
                .transpose(0, 1)
            )

        if v is not None:
            v = (
                v.contiguous()
                .view(-1, B * self.n_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        assert k.size(1) == k_N

        # query(q)/key(k)/value(v) shape: (B*h, k_N, D) [B = batch_size, h = n_heads, k_N = 12*N, D = d_k (head_dim)]

        bh, k_N, D = q.shape

        U_part = self.factor * np.ceil(np.log(k_N)).astype('int').item()  # c*ln(k_N)

        U_part = U_part if U_part < k_N else k_N

        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=U_part)
        # scores_top: [B, H, U_part/n_top, k_N]
        # index: [B, H, U_part/n_top]

        # print("scores_top shape: ", scores_top.shape)
        # print("index shape: ", index.shape)

        # get the context
        context = self._get_initial_context(v, k_N)
        # context: [B, h, k_N, D/d_k]

        # print("context shape: ", context.shape)

        # update the context with selected top_k queries
        context, attn = self._update_context(context, v, scores_top, index, k_N)
        # context shape: [B, H, k_N, D/d_k]
        # attn shape: [B, H, k_N, k_N]

        # print("context shape: ", context.shape)
        # print("attn shape: ", attn.shape)
        # raise ValueError("Enter correct prob-sparse context shape ...")

        context = context.permute(0, 2, 1, 3)  # [B, k_N, H, D/d_k]
        context = context.reshape(B, k_N, d_model)  # [B, k_N/12*N, C] : (C = H * d_k)
        # print("context shape: ", context.shape)
        # raise ValueError("Enter correct prob-sparse context shape ...")

        out = self.fc_out(context)  # [B, k_N/12*N, d_model(C)]
        # print("out shape: ", out.shape)
        # raise ValueError("Enter correct prob-sparse output shape ...")

        return out

