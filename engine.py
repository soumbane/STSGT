import torch
import torch.nn as nn
import torch.optim as optim
from models.model import *
import utils


class trainer(nn.Module):
    def __init__(self, scaler, adj, history, num_of_vertices, in_dim, hidden_dims, first_layer_embedding_size,
                 out_layer_dim, d_model, n_heads, factor, attention_dropout, output_attention, dropout,
                 forward_expansion, log, lrate, w_decay, l_decay_rate, device, activation='GLU', use_mask=True,
                 max_grad_norm=5, lr_decay=False, temporal_emb=True, spatial_emb=True, use_transformer=True,
                 use_informer=True, horizon=12, strides=12):
        """
        Trainer
        scaler: converter
        adj: local space-time matrix
        history: input time step
        num_of_vertices: num of nodes
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
        log: log
        lrate: initial learning rate
        w_decay: weight decay rate
        l_decay_rate: lr decay rate after every epoch
        device: computing device
        activation: activation function {relu, GlU}
        use_mask: Whether to use the mask matrix to optimize adj
        max_grad_norm: gradient threshold
        lr_decay: whether to use the initial learning rate decay strategy
        temporal_emb: whether to use temporal embedding vector
        spatial_emb: whether to use spatial embedding vector
        use_transformer: (bool) whether to use the ST synchronous Full-Attention or not
        use_informer: (bool) whether to use the ST synchronous Prob-sparse self-attention or not
        horizon: forecast time step
        strides: local spatio-temporal graph is constructed using these time steps, the default is 12
        """

        super(trainer, self).__init__()

        # For the STSI/STST/STSGT model (our model) ONLY
        self.model = STSGCN(
            adj=adj,
            history=history,
            num_of_vertices=num_of_vertices,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            first_layer_embedding_size=first_layer_embedding_size,
            out_layer_dim=out_layer_dim,
            d_model=d_model,
            n_heads=n_heads,
            factor=factor,
            attention_dropout=attention_dropout,
            output_attention=output_attention,
            dropout=dropout,
            forward_expansion=forward_expansion,
            activation=activation,
            use_mask=use_mask,
            temporal_emb=temporal_emb,
            spatial_emb=spatial_emb,
            use_transformer=use_transformer,
            use_informer=use_informer,
            horizon=horizon,
            strides=strides
        )

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(device)
        # print("Is Model in GPU?", next(self.model.parameters()).is_cuda)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate,
                                    weight_decay=w_decay)

        if lr_decay:
            utils.log_string(log, 'Applying Lambda Learning rate decay.')
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                                  lr_lambda=lambda epoch: l_decay_rate ** epoch)

        # Loss Function
        self.loss = utils.masked_mae

        self.scaler = scaler
        self.clip = max_grad_norm

        utils.log_string(log, "Model trainable parameters: {:,}".format(utils.count_parameters(self.model)))

        # utils.init_seed(seed=10)

    def train_model(self, input_data, real_val):
        """
        input_data: B, T, N, C
        real_val: B, T, N
        """

        self.model.train()
        self.optimizer.zero_grad()

        # B, T, N
        output = self.model(input_data)
        # print("Output Data: ", output.shape)

        # B, T, N
        predict = self.scaler.inverse_transform(output)
        # print("Predict shape: ", predict.shape)
        # print("Real Val: ", real_val.shape)

        loss = self.loss(predict, real_val, 0.0)  # for masked MAE loss

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        mae = utils.masked_mae(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()
        rmsle = utils.masked_rmsle(predict, real_val, 0.0).item()

        return loss.item(), mae, rmse, rmsle

    def eval_model(self, input_data, real_val):
        """
        input_data: B, T, N, C
        real_val:B, T, N
        """
        self.model.eval()

        # B, T, N
        output = self.model(input_data)

        # B, T, N
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real_val, 0.0)  # for masked MAE loss

        mae = utils.masked_mae(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()
        rmsle = utils.masked_rmsle(predict, real_val, 0.0).item()

        return loss.item(), mae, rmse, rmsle


