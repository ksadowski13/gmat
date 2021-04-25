import math
from typing import Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        activation: str = None,
    ) -> None:
        super().__init__()
        self._linear = nn.Linear(in_feats, out_feats)
        self._activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._linear(inputs)

        if self._activation == 'relu':
            x = F.relu(x)
        elif self._activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self._activation == 'sigmoid':
            x = torch.sigmoid(x)

        return x


class MutualMultiAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_heads: int,
        dropout_probability: float,
        message_aggregation_type: str,
        head_pooling_type: str,
        linear_projection_activation: str = None,
    ) -> None:
        super().__init__()
        self._node_in_feats = node_in_feats
        self._edge_in_feats = edge_in_feats
        self._num_heads = num_heads
        self._message_aggregation_type = message_aggregation_type
        self._head_pooling_type = head_pooling_type
        self._node_linear_projection = LinearBlock(
            node_in_feats,
            node_in_feats * num_heads,
            linear_projection_activation,
        )
        self._edge_linear_projection = LinearBlock(
            edge_in_feats,
            num_heads,
            linear_projection_activation,
        )
        self._dropout = nn.Dropout(dropout_probability)

    def _calculate_self_attention(
        self,
        linear_projection: torch.Tensor,
        in_feats: int,
    ) -> torch.Tensor:
        x = linear_projection / math.sqrt(in_feats)
        x = F.softmax(x, dim=1)

        return x

    @torch.jit.script
    def _attention_projection_script(self_attention, src_edges, dst_edges, num_heads: int, num_nodes: int, num_edges: int):
        projection = torch.zeros(
            [num_heads, num_nodes, num_nodes], dtype=torch.float32, device='cpu')

        for edge in range(num_edges):
            source = src_edges[edge]
            destination = dst_edges[edge]

            for head in range(num_heads):
                attention_score = self_attention[head][edge]

                projection[head][source][destination] = attention_score

        return projection

    def _create_node_attention_projection(
        self,
        g: dgl.DGLGraph,
        edge_self_attention: torch.Tensor,
    ) -> torch.Tensor:
        edges = g.edges()

        node_attention_projection = self._attention_projection_script(
            edge_self_attention,
            edges[0],
            edges[1],
            self._num_heads,
            g.num_nodes(),
            g.num_edges(),
        )

        return node_attention_projection

    def _calculate_message_passing(
        self,
        g: dgl.DGLGraph,
        node_linear_projection: torch.Tensor,
        node_attention_projection: torch.Tensor,
    ) -> torch.Tensor:
        if self._message_aggregation_type == 'sum':
            x = node_attention_projection
        elif self._message_aggregation_type == 'mean':
            degree_inv = torch.linalg.inv(torch.diag(g.in_degrees().float()))

            x = degree_inv @ node_attention_projection
        elif self._message_aggregation_type == 'gcn':
            adjacency = g.adj(ctx=self._device.device).to_dense()

            degree_inv_sqrt = torch.sqrt(torch.linalg.inv(
                torch.diag(g.in_degrees().float())))
            adjacency_inv_sqrt = torch.sqrt(torch.linalg.inv(adjacency))

            x = degree_inv_sqrt @ node_attention_projection * \
                adjacency_inv_sqrt @ degree_inv_sqrt

        message_passing = x @ node_linear_projection

        return message_passing

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> torch.Tensor:
        node_linear_projection = self._node_linear_projection(node_inputs)
        node_linear_projection = node_linear_projection.view(
            self._num_heads, -1, self._node_in_feats)

        edge_linear_projection = self._edge_linear_projection(edge_inputs)
        edge_linear_projection = edge_linear_projection.view(
            self._num_heads, -1, 1)

        edge_self_attention = self._calculate_self_attention(
            edge_linear_projection, self._edge_in_feats)

        node_attention_projection = self._create_node_attention_projection(
            g, edge_self_attention)

        message_passing = self._calculate_message_passing(
            g, node_linear_projection, node_attention_projection)
        message_passing = self._dropout(message_passing)

        if self._head_pooling_type == 'sum':
            message_passing = message_passing.sum(dim=-3)
        elif self._head_pooling_type == 'mean':
            message_passing = message_passing.mean(dim=-3)

        return message_passing


class MutualAttentionLayer(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        num_heads: int,
        long_residual: bool,
        dropout_probability: float,
        message_aggregation_type: str,
        head_pooling_type: str,
        normalization_type: str,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
    ) -> None:
        super().__init__()
        self._long_residual = long_residual
        self._mutual_multi_attention_head = MutualMultiAttentionHead(
            node_in_feats,
            edge_in_feats,
            num_heads,
            dropout_probability,
            message_aggregation_type,
            head_pooling_type,
            linear_projection_activation,
        )
        self._linear_embedding = LinearBlock(
            node_in_feats, node_out_feats, linear_embedding_activation)

        if normalization_type == 'layer':
            self._normalization_1 = nn.LayerNorm(node_in_feats)
            self._normalization_2 = nn.LayerNorm(node_out_feats)
        elif normalization_type == 'batch':
            self._normalization_1 = nn.BatchNorm1d(node_in_feats)
            self._normalization_2 = nn.BatchNorm1d(node_out_feats)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embedding = self._mutual_multi_attention_head(
            g, node_inputs, edge_inputs)

        if self._long_residual:
            node_embedding += node_inputs

        node_embedding = self._normalization_1(node_embedding)

        node_embedding = self._linear_embedding(node_embedding)

        node_embedding = self._normalization_2(node_embedding)

        return node_embedding


class GraphMutualAttentionNetwork(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_hidden_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        num_layers: int,
        num_heads: int,
        long_residual: bool,
        dropout_probability: float,
        message_aggregation_type: str,
        head_pooling_type: str,
        readout_pooling_type: str,
        normalization_type: str,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
        linear_readout_activation: str = None,
    ) -> None:
        super().__init__()
        self._node_out_feats = node_out_feats
        self._num_layers = num_layers
        self._mutual_attention_layers = self._create_mutual_attention_layers(
            node_in_feats,
            node_hidden_feats,
            node_out_feats,
            edge_in_feats,
            num_layers,
            num_heads,
            long_residual,
            dropout_probability,
            message_aggregation_type,
            head_pooling_type,
            normalization_type,
            linear_projection_activation,
            linear_embedding_activation,
        )
        self._linear_readout = LinearBlock(
            node_out_feats, 1, linear_readout_activation)

        if readout_pooling_type == 'sum':
            self._readout_pooling = dgl.nn.pytorch.SumPooling()
        elif readout_pooling_type == 'mean':
            self._readout_pooling = dgl.nn.pytorch.AvgPooling()
        elif readout_pooling_type == 'attention':
            pass

    def _create_mutual_attention_layers(
        self,
        node_in_feats: int,
        node_hidden_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        num_layers: int,
        num_heads: int,
        long_residual: bool,
        dropout_probability: float,
        message_aggregation_type: str,
        head_pooling_type: str,
        normalization_type: str,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
    ) -> nn.ModuleList:
        mutual_attention_layers = nn.ModuleList()

        if num_layers > 1:
            mutual_attention_layers.append(MutualAttentionLayer(
                node_in_feats,
                node_hidden_feats,
                edge_in_feats,
                num_heads,
                long_residual,
                dropout_probability,
                message_aggregation_type,
                head_pooling_type,
                normalization_type,
                linear_projection_activation,
                linear_embedding_activation,
            ))

            for _ in range(num_layers - 2):
                mutual_attention_layers.append(MutualAttentionLayer(
                    node_hidden_feats,
                    node_hidden_feats,
                    edge_in_feats,
                    num_heads,
                    long_residual,
                    dropout_probability,
                    message_aggregation_type,
                    head_pooling_type,
                    normalization_type,
                    linear_projection_activation,
                    linear_embedding_activation,
                ))

            mutual_attention_layers.append(MutualAttentionLayer(
                node_hidden_feats,
                node_out_feats,
                edge_in_feats,
                num_heads,
                long_residual,
                dropout_probability,
                message_aggregation_type,
                head_pooling_type,
                normalization_type,
                linear_projection_activation,
                linear_embedding_activation,
            ))
        else:
            mutual_attention_layers.append(MutualAttentionLayer(
                node_in_feats,
                node_out_feats,
                edge_in_feats,
                num_heads,
                long_residual,
                dropout_probability,
                message_aggregation_type,
                head_pooling_type,
                normalization_type,
                linear_projection_activation,
                linear_embedding_activation,
            ))

        return mutual_attention_layers

    def forward(
        self,
        g: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> torch.Tensor:
        node_embedding = node_inputs

        for mutual_attention_layer in self._mutual_attention_layers:
            node_embedding = mutual_attention_layer(
                g, node_embedding, edge_inputs)

        node_embedding = self._readout_pooling(g, node_embedding)

        readout = self._linear_readout(node_embedding)

        return readout
