from typing import Optional

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class STGraphEncoder(nn.Module):
    def __init__(
            self,
            d_model: int = 8,
            n_heads: int = 1,
            factor: int = 10,
            attention_dropout: float = 0.1,
            output_attention: bool = False,
            dropout: float = 0.1,
            forward_expansion: int = 64,
            use_informer: bool = True
    ) -> None:
        """
        Spatio-Temporal Synchronous Transformer Module

        d_model: The embedding dimension of Query, Key, Value for self-attention
        n_heads: The number of heads for calculating multi-head self attention
        factor: The amount of self-attentions (top queries) to be selected
        attention_dropout: The dropout rate for the self-attention
        output_attention: Whether to output self-attentions or not
        dropout: Dropout Ratio after multi-head self-attention
        forward_expansion: The dimension of the hidden layer of MLP after multi-head self attention
        use_informer: Whether to use the prob-sparse self-attention of the Informer
        """

        super(STGraphEncoder, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention

        if use_informer:
            from models.attn import STSProbSparseSelfAttention

            self.self_attention = STSProbSparseSelfAttention(d_model=d_model,
                                                             n_heads=n_heads,
                                                             factor=factor,
                                                             dropout=attention_dropout,
                                                             output_attention=output_attention,
                                                             self_attention=True
                                                             )
        else:
            from models.attn import STSFullSelfAttention

            self.self_attention = STSFullSelfAttention(d_model=d_model,
                                                       n_heads=n_heads,
                                                       dropout=attention_dropout,
                                                       self_attention=True
                                                       )

        self.norm1 = nn.LayerNorm(d_model)

        self.fc1 = nn.Linear(d_model, forward_expansion * d_model)

        self.fc2 = nn.Linear(forward_expansion * d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        query: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        return out: [12*N(Spatio-Temporal), B(Batch-Size), C(d_model)]
        """

        self_attention = self.self_attention(
            query=self.norm1(query),
            key=self.norm1(query),
            value=self.norm1(query),
        )
        # attention: [B, 12*N, C]
        # print("Attention shape before mask: ", attention.shape)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self_attention) + query.permute(1, 0, 2)

        out = self.dropout(self.fc2(self.dropout(torch.relu(self.fc1(self.norm1(x)))))) + x

        return out.permute(1, 0, 2)  # (12*N, B, C)


class gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
        """
        Graph Convolution Module

        adj: adjacency graph
        in_dim: input dimension
        out_dim: output dimension
        num_vertices: number of nodes
        activation: activation method {'relu','GLU'}
        """

        super(gcn_operation, self).__init__()

        self.adj = adj  # [12*N, 12*N]
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_vertices = num_vertices
        self.activation = activation

        assert self.activation in {'GLU', 'relu'}

        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):
        """
        x: (12*N, B, Cin)
        mask:(12*N, 12*N)
        return: (12*N, B, Cout)
        """
        adj = self.adj
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 12*N, B, Cin

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 12*N, B, 2*Cout
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 12*N, B, Cout

            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs

            return out  # 12*N, B, Cout

        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 12*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, d_model, n_heads, factor, attention_dropout,
                 output_attention, dropout, forward_expansion, activation='GLU', use_transformer=True,
                 use_informer=True):
        """
        adj: adjacency matrix
        in_dim: input dimension
        out_dims: list output dimensions of each graph convolution
        num_of_vertices: number of nodes
        d_model: The dimension of Query, Key, Value for self-attention
        n_heads: The number of heads for calculating multi-head self attention
        factor: The amount of self-attentions (top queries) to be selected
        attention_dropout: The dropout rate for the self-attention
        output_attention: Whether to output self-attentions or not
        dropout: Dropout Ratio after multi-head self-attention
        forward_expansion: The dimension of the hidden layer of MLP after multi-head self attention
        activation: activation method {'relu','GLU'}
        use_transformer: (bool) whether to use the ST synchronous Full-Attention or not
        use_informer: (bool) whether to use the ST synchronous Prob-sparse self-attention or not
        """

        super(STSGCM, self).__init__()

        self.adj = adj
        self.in_dim = in_dim
        self.out_dims = out_dims

        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.dropout = dropout
        self.forward_expansion = forward_expansion

        self.use_transformer = use_transformer
        self.use_informer = use_informer

        self.gcn_operations = nn.ModuleList()

        self.gcn_operations.append(
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )

        for i in range(1, len(self.out_dims)):
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],
                    out_dim=self.out_dims[i],
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

        self.ST_GraphEncoders = nn.ModuleList()

        for i in range(len(self.out_dims)):
            self.ST_GraphEncoders.append(
                STGraphEncoder(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    factor=self.factor,
                    attention_dropout=self.attention_dropout,
                    output_attention=self.output_attention,
                    dropout=self.dropout,
                    forward_expansion=self.forward_expansion,
                    use_informer=self.use_informer
                )
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (12*N, B, Cin)
        mask: (12*N, 12*N)
        return: (12*N, B, Cout)
        """

        # print("Shape of x: ", x.shape)
        # k_N, B, d_model = x.shape

        input_transformer = x

        if self.use_transformer:
            for i in range(len(self.out_dims)):
                transformer_temp = self.ST_GraphEncoders[i](input_transformer, mask)  # [12*N, B, C]

                transformer_temp = self.gcn_operations[i](transformer_temp, mask)  # [12*N, B, C]

                # Adding Skip Connections
                input_transformer = transformer_temp + input_transformer

            input_transformer = input_transformer + x

            out = input_transformer  # [12*N, B, C]

            del input_transformer, transformer_temp

            return out  # [12*N, B, C]

        else:
            raise NotImplementedError


class STSGCL(nn.Module):
    def __init__(self, adj, history, num_of_vertices, in_dim, out_dims, d_model, n_heads, factor, attention_dropout,
                 output_attention, dropout, forward_expansion, strides=7, activation='GLU', temporal_emb=True,
                 spatial_emb=True, use_transformer=True, use_informer=True):
        """
        adj: adjacency matrix
        history: input time step
        in_dim: input dimension
        out_dims: list output dimensions of each graph convolution
        d_model: The dimension of Query, Key, Value for self-attention
        n_heads: The number of heads for calculating multi-head self attention
        factor: The amount of self-attentions (top queries) to be selected
        attention_dropout: The dropout rate for the self-attention
        output_attention: Whether to output self-attentions or not
        dropout: Dropout Ratio after multi-head self-attention
        forward_expansion: The dimension of the hidden layer of MLP after multi-head self attention
        strides: local spatio-temporal graph is constructed using these time steps, the default is 12
        num_of_vertices: number of nodes
        activation: activation method {'relu','GLU'}
        temporal_emb: add temporal position embedding vector
        spatial_emb: add spatial position embedding vector
        use_transformer: (bool) whether to use the ST synchronous Full-Attention or not
        use_informer: (bool) whether to use the ST synchronous Prob-sparse self-attention or not
        """
        super(STSGCL, self).__init__()
        self.adj = adj
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.out_dims = out_dims
        self.num_of_vertices = num_of_vertices

        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        # The following is needed for the ST Self-Attentions
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.dropout = dropout
        self.forward_expansion = forward_expansion

        self.use_transformer = use_transformer
        self.use_informer = use_informer

        self.STSGCMS = STSGCM(adj=self.adj,
                              in_dim=self.in_dim,
                              out_dims=self.out_dims,
                              num_of_vertices=self.num_of_vertices,
                              activation=self.activation,
                              d_model=self.d_model,
                              n_heads=self.n_heads,
                              factor=self.factor,
                              attention_dropout=self.attention_dropout,
                              output_attention=self.output_attention,
                              dropout=self.dropout,
                              forward_expansion=self.forward_expansion,
                              use_transformer=self.use_transformer,
                              use_informer=self.use_informer
                              )

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim),
                                                   requires_grad=True)
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim),
                                                  requires_grad=True)
            # 1, 1, N, Cin

        self._reset_parameters()

        self.parallel_1D_conv = nn.Conv2d(in_channels=self.in_dim,
                                          out_channels=self.out_dims[0],
                                          kernel_size=(1, 2),
                                          dilation=(1, 2)
                                          )

    def _reset_parameters(self):
        if self.temporal_emb:
            # nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
            nn.init.xavier_uniform_(self.temporal_embedding, gain=1 / math.sqrt(2))

        if self.spatial_emb:
            # nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)
            nn.init.xavier_uniform_(self.spatial_embedding, gain=1 / math.sqrt(2))

    def forward(self, x, mask=None):
        """
        x: B, T, N, Cin (T=12), Cin=first_layer_embedding_size
        mask: (12*N, 12*N)
        return: B, T, N, Cout (T=12)
        """
        # print("STSGCL Input shape: ", x.shape)  # (B, T, N, C)
        # print("Mask shape: ", mask.shape)
        # raise ValueError("Enter Correct Mask Value .... ")

        if self.temporal_emb:
            x = x + self.temporal_embedding

            # print("Temporal Embedding Shape: ", self.temporal_embedding.shape)  # (1, T, 1, C)
            # print("Temporal Embeddings: ", self.temporal_embedding[:, :, :, :5])
            # print("X Shape: ", x.shape)  # (B, T, N, C)

        if self.spatial_emb:
            x = x + self.spatial_embedding

            # print("Spatial Embedding Shape: ", self.spatial_embedding.shape)  # (1, 1, N, C)
            # print("Spatial Embeddings: ", self.spatial_embedding[:, :, :5, :5])
            # print("X Shape: ", x.shape)  # (B, T, N, C)

        # X shape: (B, T, N, C)
        # print("Shape of X (input of STSGCL): ", x.shape)

        # The following is the ST Self-Attention and GCN part
        # X shape: (B, T, N, C)
        B = x.shape[0]
        T = x.shape[1]

        t = x[:, :self.strides, :, :]  # (B, 12, N, Cin)

        # (B, 12*N, Cin)
        t = torch.reshape(t, shape=[B, self.strides * self.num_of_vertices, self.in_dim])

        t = self.STSGCMS(t.permute(1, 0, 2), mask)  # (12*N, B, Cin) -> (12*N, B, Cout)
        # print("Shape of t: ", t.shape)

        # (12*N, B, Cout) -> (12, N, B, Cout)
        t1 = t.view(T, self.num_of_vertices, B, -1)
        # print("Shape of t1: ", t1.shape)

        out1 = t1.permute(2, 0, 1, 3)  # [B, T, N, Cout]
        out = out1 + x

        del B, T, t, t1, out1

        # print("Output of STSGCL shape: ", out.shape)

        return out  # [B, T, N, Cout]


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128, horizon=7):
        """
        The prediction layer

        num_of_vertices: number of nodes
        history: input time step
        in_dim: input dimension
        hidden_dim: middle layer dimension
        horizon: prediction time step
        """

        super(output_layer, self).__init__()

        self.num_of_vertices = num_of_vertices
        self.history = history
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.output_conv_1 = nn.Conv2d(in_channels=self.in_dim * self.history,
                                       out_channels=self.hidden_dim,
                                       kernel_size=(1, 1),
                                       bias=True
                                       )

        self.output_conv_2 = nn.Conv2d(in_channels=self.hidden_dim,
                                       out_channels=self.horizon,
                                       kernel_size=(1, 1),
                                       bias=True
                                       )

    def forward(self, x):
        """
        x: (B, Tin, N, Cin)
        return: (B, Tout, N)
        """
        batch_size = x.shape[0]
        # print("Output Layer Input Shape: ", x.shape)

        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin
        # x = x.permute(0, 3, 2, 1)  # B, Cin, N, Tin
        # print("Output Layer Input Shape after permute and reshape: ", x.shape)

        x = x.reshape(batch_size, self.num_of_vertices, -1)
        x = x.permute(0, 2, 1).unsqueeze(3)
        out1 = F.relu(self.output_conv_1(F.relu(x)))
        # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, Tin * Cin, N, 1) -> (B, hidden, N, 1)
        # print("Output Layer Shape after first conv2d: ", out1.shape)

        out2 = self.output_conv_2(out1).squeeze(3)  # (B, hidden, N, 1) -> (B, horizon, N, 1) -> (B, horizon, N)

        # print("Output Layer Shape after second conv2d: ", out2.shape)

        out = out2

        del out1, out2
        # print("Output Layer output shape: ", out.shape)

        return out  # B, horizon, N


class STSGCN(nn.Module):
    def __init__(self, adj, history, num_of_vertices, in_dim, hidden_dims, first_layer_embedding_size,
                 out_layer_dim, d_model, n_heads, factor, attention_dropout, output_attention, dropout,
                 forward_expansion, activation='GLU', use_mask=True, temporal_emb=True, spatial_emb=True,
                 use_transformer=False, use_informer=True, horizon=7, strides=12):
        """
        adj: local space-time matrix
        history: input time step
        num_of_vertices: number of nodes
        in_dim: input dimension
        hidden_dims: lists, the convolution operation dimension of each STSGCL layer in the middle
        first_layer_embedding_size: the dimension of the first input layer
        out_layer_dim: output module middle layer dimension
        d_model: The dimension of Query, Key, Value for self-attention
        n_heads: The number of heads for calculating multi-head self attention
        factor: The amount of self-attentions (top queries) to be selected
        attention_dropout: The dropout rate for the self-attention
        output_attention: Whether to output self-attentions or not
        dropout: Dropout Ratio after multi-head self-attention
        forward_expansion: The dimension of the hidden layer of MLP after multi-head self attention
        activation: activation function {relu, GlU}
        use_mask: Whether to use the mask matrix to optimize adj
        temporal_emb: Whether to use temporal embedding vector
        spatial_emb: Whether to use spatial embedding vector
        horizon: prediction time step
        strides: local spatio-temporal graph is constructed using these time steps, the default is 12
        use_transformer: (bool) whether to use the ST synchronous Full-Attention or not
        use_informer: (bool) whether to use the ST synchronous Prob-sparse self-attention or not
        """

        super(STSGCN, self).__init__()

        self.adj = adj
        self.history = history
        self.num_of_vertices = num_of_vertices
        self.hidden_dims = hidden_dims
        self.out_layer_dim = out_layer_dim

        # The following is needed for the ST Self-Attentions
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor
        self.attention_dropout = attention_dropout
        self.output_attention = output_attention
        self.dropout = dropout
        self.forward_expansion = forward_expansion

        self.use_transformer = use_transformer
        self.use_informer = use_informer

        self.activation = activation
        self.use_mask = use_mask

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        self.horizon = horizon
        self.strides = strides

        self.input_conv_1 = nn.Conv2d(in_channels=in_dim,
                                      out_channels=first_layer_embedding_size,
                                      kernel_size=(1, 1)
                                      )

        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj,
                history=self.history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0],
                d_model=self.d_model,
                n_heads=self.n_heads,
                factor=self.factor,
                attention_dropout=self.attention_dropout,
                output_attention=self.output_attention,
                dropout=self.dropout,
                forward_expansion=self.forward_expansion,
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb,
                use_transformer=self.use_transformer,
                use_informer=self.use_informer
            )
        )

        in_dim = self.hidden_dims[0][-1]

        for idx, hidden_list in enumerate(self.hidden_dims):
            if idx == 0:
                continue
            self.STSGCLS.append(
                STSGCL(
                    adj=self.adj,
                    history=self.history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,
                    out_dims=hidden_list,
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    factor=self.factor,
                    attention_dropout=self.attention_dropout,
                    output_attention=self.output_attention,
                    dropout=self.dropout,
                    forward_expansion=self.forward_expansion,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb,
                    use_transformer=self.use_transformer,
                    use_informer=self.use_informer
                )
            )

            in_dim = hidden_list[-1]

        self.predictLayer = output_layer(num_of_vertices=self.num_of_vertices,
                                         history=self.history,
                                         in_dim=in_dim,
                                         hidden_dim=out_layer_dim,
                                         horizon=horizon
                                         )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask, requires_grad=True)
        else:
            self.mask = None

    def forward(self, x):
        """
        x: B, Tin, N, Cin
        return: B, Tout, N
        """
        # print("Original Input shape: ", x.shape)  # B, Tin, N, Cin

        x = torch.relu(self.input_conv_1(x.permute(0, 3, 2, 1)))  # B, C_first_layer_embedding_size, N, Tin
        # print("After first Input conv2d: ", x.shape)

        x = x.permute(0, 3, 2, 1)  # B, Tin, N, C_first_layer_embedding_size
        # print("After permute: ", x.shape)

        for model in self.STSGCLS:
            x = model(x, self.mask)
        # (B, T, N, Cout)
        # print(f'Tensor Shape after {len(self.STSGCLS)} STSGCL layers is: {x.shape}')

        out = self.predictLayer(x)  # (B, T, N)

        # print("Model Output shape: ", out.shape)

        return out


