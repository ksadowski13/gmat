import math
from typing import Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        normalization: str = None,
        activation: str = None,
    ):
        super().__init__()
        self._linear = nn.Linear(in_feats, out_feats)

        if normalization is not None:
            if normalization == 'batch':
                self._normalization = nn.BatchNorm1d(out_feats)
            elif normalization == 'layer':
                self._normalization = nn.LayerNorm(out_feats)
        else:
            self._normalization = None

        if activation is not None:
            if activation == 'relu':
                self._activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self._activation = nn.LeakyReLU()
            elif activation == 'sigmoid':
                self._activation = nn.Sigmoid()
        else:
            self._activation = None

    def forward(self, inputs: torch.Tensor):
        x = self._linear(inputs)

        if self._normalization is not None:
            x = self._normalization(x)

        if self._activation is not None:
            x = self._activation(x)

        return x


class BilinearBlock(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        out_feats: int,
        activation: str = None,
    ):
        super().__init__()
        self._bilinear = nn.Bilinear(node_in_feats, edge_in_feats, out_feats)

        if activation is not None:
            if activation == 'relu':
                self._activation = nn.ReLU()
            elif activation == 'softplus':
                self._activation == nn.Softplus()
        else:
            self._activation = None

    def forward(
        self,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self._bilinear(node_inputs, edge_inputs)

        if self._activation is not None:
            x = self._activation(x)

        return x


class MessageProjection(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_heads: int,
        message_func: str,
        reduce_func: str,
        node_activation: str = None,
        edge_activation: str = None,
    ):
        super().__init__()
        self._node_in_feats = node_in_feats
        self._num_heads = num_heads
        self._node_linear = LinearBlock(
            node_in_feats,
            node_in_feats * num_heads,
            activation=node_activation,
        )
        self._edge_linear = LinearBlock(
            edge_in_feats,
            num_heads,
            activation=edge_activation,
        )

        if message_func == 'add':
            self._message_func = fn.u_add_e('projection', 'weight', 'message')
        elif message_func == 'sub':
            self._message_func = fn.u_sub_e('projection', 'weight', 'message')
        elif message_func == 'mul':
            self._message_func = fn.u_mul_e('projection', 'weight', 'message')
        elif message_func == 'div':
            self._message_func = fn.u_div_e('projection', 'weight', 'message')

        if reduce_func == 'sum':
            self._reduce_func = fn.sum('message', 'projection')
        elif reduce_func == 'mean':
            self._reduce_func = fn.mean('message', 'projection')

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g.ndata['projection'] = self._node_linear(node_inputs)
        g.ndata['projection'] = g.ndata['projection'].view(
            -1, self._num_heads, self._node_in_feats)

        g.edata['weight'] = self._edge_linear(edge_inputs)
        g.edata['weight'] = g.edata['weight'].view(-1, self._num_heads, 1)

        g.update_all(
            message_func=self._message_func,
            reduce_func=self._reduce_func,
        )

        node_projection = g.ndata.pop('projection')
        edge_projection = g.edata.pop('weight')

        return node_projection, edge_projection


class LinearProjection(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_heads: int,
        activation: str = None,
    ):
        super().__init__()
        self._node_in_feats = node_in_feats
        self._edge_in_feats = edge_in_feats
        self._num_heads = num_heads
        self._node_linear = LinearBlock(
            node_in_feats,
            node_in_feats * num_heads,
            activation=activation,
        )
        self._edge_linear = LinearBlock(
            edge_in_feats,
            edge_in_feats * num_heads,
            activation=activation,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_projection = self._node_linear(node_inputs)
        node_projection = node_projection.view(
            -1, self._num_heads, self._node_in_feats)

        edge_projection = self._edge_linear(edge_inputs)
        edge_projection = edge_projection.view(
            -1, self._num_heads, self._edge_in_feats)

        return node_projection, edge_projection


class MutualMultiAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_heads: int,
        message_func: str,
        reduce_func: str,
        head_pooling_func: str,
        weight_activation: str = None,
        projection_activation: str = None,
    ):
        super().__init__()
        self._node_scale_const = math.sqrt(node_in_feats)
        self._edge_scale_const = math.sqrt(edge_in_feats)
        self._head_pooling_func = head_pooling_func
        self._query_linear = MessageProjection(
            node_in_feats,
            edge_in_feats,
            num_heads,
            message_func,
            reduce_func,
            projection_activation,
            weight_activation,
        )
        self._key_linear = MessageProjection(
            node_in_feats,
            edge_in_feats,
            num_heads,
            message_func,
            reduce_func,
            projection_activation,
            weight_activation,
        )
        self._value_linear = LinearProjection(
            node_in_feats,
            edge_in_feats,
            num_heads,
            projection_activation,
        )

    def _calculate_attention_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scale_const: float,
    ):
        attention_score = query @ torch.transpose(key, -1, -2)
        attention_score = torch.exp(attention_score / scale_const).clamp(-5, 5)
        attention_score = F.softmax(attention_score, dim=-1)

        return attention_score

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_query, edge_query = self._query_linear(
            g, node_inputs, edge_inputs)
        node_key, edge_key = self._key_linear(g, node_inputs, edge_inputs)
        node_value, edge_value = self._value_linear(
            g, node_inputs, edge_inputs)

        node_attention = self._calculate_attention_score(
            node_query, node_key, self._node_scale_const)
        edge_attention = self._calculate_attention_score(
            edge_query, edge_key, self._edge_scale_const)

        node_embedding = node_attention @ node_value
        edge_embedding = edge_attention @ edge_value

        if self._head_pooling_func == 'sum':
            node_embedding = node_embedding.sum(-2)
            edge_embedding = edge_embedding.sum(-2)
        elif self._head_pooling_func == 'mean':
            node_embedding = node_embedding.mean(-2)
            edge_embedding = edge_embedding.mean(-2)

        return node_embedding, edge_embedding


class MutualAttentionTransformerLayer(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_hidden_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_hidden_feats: int,
        edge_out_feats: int,
        num_heads: int,
        message_func: str,
        reduce_func: str,
        head_pooling_func: str,
        residual: bool,
        dropout: float,
        embedding_normalization: str = None,
        weight_activation: str = None,
        projection_activation: str = None,
        embedding_activation: str = None,
    ):
        super().__init__()
        self._residual = residual
        self._mutual_multi_attention_head = MutualMultiAttentionHead(
            node_in_feats,
            edge_in_feats,
            num_heads,
            message_func,
            reduce_func,
            head_pooling_func,
            weight_activation,
            projection_activation,
        )
        self._node_embedding_dropout_1 = nn.Dropout(dropout)
        self._node_embedding_dropout_2 = nn.Dropout(dropout)
        self._edge_embedding_dropout_1 = nn.Dropout(dropout)
        self._edge_embedding_dropout_2 = nn.Dropout(dropout)
        self._node_embedding_linear_1 = LinearBlock(
            node_in_feats,
            node_hidden_feats,
            normalization=embedding_normalization,
            activation=embedding_activation,
        )
        self._node_embedding_linear_2 = LinearBlock(
            node_hidden_feats,
            node_out_feats,
            normalization=embedding_normalization,
            activation=embedding_activation,
        )
        self._edge_embedding_linear_1 = LinearBlock(
            edge_in_feats,
            edge_hidden_feats,
            normalization=embedding_normalization,
            activation=embedding_activation,
        )
        self._edge_embedding_linear_2 = LinearBlock(
            edge_hidden_feats,
            edge_out_feats,
            normalization=embedding_normalization,
            activation=embedding_activation,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embedding, edge_embedding = self._mutual_multi_attention_head(
            g, node_inputs, edge_inputs)

        node_embedding = self._node_embedding_dropout_1(node_embedding)
        edge_embedding = self._edge_embedding_dropout_1(edge_embedding)

        if self._residual:
            node_embedding += node_inputs
            edge_embedding += edge_inputs

        node_embedding = self._node_embedding_linear_1(node_embedding)
        edge_embedding = self._edge_embedding_linear_1(edge_embedding)

        node_embedding = self._node_embedding_dropout_2(node_embedding)
        edge_embedding = self._edge_embedding_dropout_2(edge_embedding)

        node_embedding = self._node_embedding_linear_2(node_embedding)
        edge_embedding = self._edge_embedding_linear_2(edge_embedding)

        return node_embedding, edge_embedding


class GraphMututalAttentionTransformer(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_hidden_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_hidden_feats: int,
        edge_out_feats: int,
        num_layers: int,
        num_heads: int,
        message_func: str,
        reduce_func: str,
        head_pooling_func: str,
        readout_pooling_func: str,
        residual: bool,
        dropout: float,
        embedding_normalization: str = None,
        weight_activation: str = None,
        projection_activation: str = None,
        embedding_activation: str = None,
        readout_activation: str = None,
    ):
        super().__init__()
        self._node_out_feats = node_out_feats
        self._edge_out_feats = edge_out_feats
        self._transformer_layers = self._create_transformer_layers(
            node_in_feats,
            node_hidden_feats,
            node_out_feats,
            edge_in_feats,
            edge_hidden_feats,
            edge_out_feats,
            num_layers,
            num_heads,
            message_func,
            reduce_func,
            head_pooling_func,
            residual,
            dropout,
            embedding_normalization,
            weight_activation,
            projection_activation,
            embedding_activation,
        )

        if readout_pooling_func == 'sum':
            self._readout_pooling = dgl.nn.pytorch.SumPooling()
        elif readout_pooling_func == 'mean':
            self._readout_pooling = dgl.nn.pytorch.AvgPooling()

        self._bilinear = BilinearBlock(
            node_out_feats,
            edge_out_feats,
            1,
            readout_activation,
        )

    def _create_transformer_layers(
        self,
        node_in_feats: int,
        node_hidden_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_hidden_feats: int,
        edge_out_feats: int,
        num_layers: int,
        num_heads: int,
        message_func: str,
        reduce_func: str,
        head_pooling_func: str,
        residual: bool,
        dropout: float,
        embedding_normalization: str = None,
        weight_activation: str = None,
        projection_activation: str = None,
        embedding_activation: str = None,
    ) -> nn.ModuleList:
        transformer_layers = nn.ModuleList()

        if num_layers > 1:
            transformer_layers.append(MutualAttentionTransformerLayer(
                node_in_feats,
                node_in_feats * node_hidden_feats,
                node_hidden_feats,
                edge_in_feats,
                edge_in_feats * edge_hidden_feats,
                edge_hidden_feats,
                num_heads,
                message_func,
                reduce_func,
                head_pooling_func,
                residual,
                dropout,
                embedding_normalization,
                weight_activation,
                projection_activation,
                embedding_activation,
            ))

            for _ in range(num_layers - 2):
                transformer_layers.append(MutualAttentionTransformerLayer(
                    node_hidden_feats,
                    node_hidden_feats * node_hidden_feats,
                    node_hidden_feats,
                    edge_hidden_feats,
                    edge_hidden_feats * edge_hidden_feats,
                    edge_hidden_feats,
                    num_heads,
                    message_func,
                    reduce_func,
                    head_pooling_func,
                    residual,
                    dropout,
                    embedding_normalization,
                    weight_activation,
                    projection_activation,
                    embedding_activation,
                ))

            transformer_layers.append(MutualAttentionTransformerLayer(
                node_hidden_feats,
                node_hidden_feats * node_out_feats,
                node_out_feats,
                edge_hidden_feats,
                edge_hidden_feats * edge_out_feats,
                edge_out_feats,
                num_heads,
                message_func,
                reduce_func,
                head_pooling_func,
                residual,
                dropout,
                embedding_normalization,
                weight_activation,
                projection_activation,
                embedding_activation,
            ))

        return transformer_layers

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> torch.Tensor:
        node_embedding = node_inputs
        edge_embedding = edge_inputs

        for transformer_layer in self._transformer_layers:
            node_embedding, edge_embedding = transformer_layer(
                g, node_embedding, edge_embedding)

        node_embedding = self._readout_pooling(g, node_embedding)
        edge_embedding = self._readout_pooling(g.line_graph(backtracking=False), edge_embedding)

        readout = self._bilinear(node_embedding, edge_embedding)

        return readout
