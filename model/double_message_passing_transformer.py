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
        normalization: str = None,
        activation: str = None,
    ) -> None:
        super().__init__()
        self._linear = nn.Linear(in_feats, out_feats)
        
        if normalization == 'batch':
            self._normalization = nn.BatchNorm1d(out_feats)
        elif normalization == 'layer':
            self._normalization = nn.LayerNorm(out_feats)
        elif normalization == None:
            self._normalization = None

        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self._activation = nn.LeakyReLU()
        elif activation == None:
            self._activation == None


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
    ) -> None:
        super().__init__()
        self._bilinear = nn.Bilinear(node_in_feats, edge_in_feats, out_feats)

        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'softplus':
            self._activation = nn.Softplus()
        elif activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'softmax':
            self._activation = nn.Softmax(dim=1)
        elif activation == None:
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
        self._device = nn.Parameter(torch.empty(0))
        self._node_query_key_linear_2 = LinearBlock(
            node_in_feats,
            num_heads * 4,
            activation=linear_projection_activation,
        )
        self._node_query_key_linear_2 = LinearBlock(
            num_heads * 4,
            num_heads,
            activation=linear_projection_activation,
        )
        self._node_value_linear_1 = LinearBlock(
            node_in_feats,
            num_heads * node_in_feats * 4,
            activation=linear_projection_activation,
        )
        self._node_value_linear_2 = LinearBlock(
            num_heads * node_in_feats * 4,
            num_heads * node_in_feats,
            activation=linear_projection_activation,
        )
        self._edge_query_key_linear_1 = LinearBlock(
            edge_in_feats, 
            num_heads * 4, 
            activation=linear_projection_activation,
        )
        self._edge_query_key_linear_2 = LinearBlock(
            num_heads * 4, 
            num_heads, 
            activation=linear_projection_activation,
        )
        self._edge_value_linear_1 = LinearBlock(
            edge_in_feats,
            num_heads * edge_in_feats * 4,
            activation=linear_projection_activation,
        )
        self._edge_value_linear = LinearBlock(
            num_heads * edge_in_feats * 4,
            num_heads * edge_in_feats,
            activation=linear_projection_activation,
        )
        self._node_dropout = nn.Dropout(dropout_probability)
        self._edge_dropout = nn.Dropout(dropout_probability)

    def _calculate_self_attention(
        self,
        query: torch.Tensor,
        in_feats: int,
    ) -> torch.Tensor:
        x = query / math.sqrt(in_feats)
        x = F.softmax(x, dim=1)

        return x

    @torch.jit.script
    def _node_attention_script(
        edge_self_attention: torch.Tensor,
        source_nodes: torch.Tensor,
        destination_nodes: torch.Tensor,
        num_heads: int,
        num_nodes: int,
        num_edges: int
    ) -> torch.Tensor:
        attention_projection = torch.zeros(
            [num_heads, num_nodes, num_nodes],
            dtype=torch.float32,
            device='cuda',
        )

        for edge in range(num_edges):
            source = source_nodes[edge]
            destination = destination_nodes[edge]

            for head in range(num_heads):
                attention_score = edge_self_attention[head][edge]

                attention_projection[head][source][destination] = attention_score

        return attention_projection

    def _create_node_attention_projection(
        self,
        g: dgl.DGLGraph,
        edge_self_attention: torch.Tensor,
    ) -> torch.Tensor:
        edges = g.edges()

        node_attention_projection = self._node_attention_script(
            edge_self_attention,
            edges[0],
            edges[1],
            self._num_heads,
            g.num_nodes(),
            g.num_edges(),
        )

        return node_attention_projection

    @torch.jit.script
    def _edge_attention_script(
        node_self_attention: torch.Tensor,
        source_lg_nodes: torch.Tensor,
        destination_lg_nodes: torch.Tensor,
        destination_g_nodes: torch.Tensor,
        num_heads: int,
        num_g_edges: int,
        num_lg_edges: int,
    ):
        attention_projection = torch.zeros(
            [num_heads, num_g_edges, num_g_edges],
            dtype=torch.float32,
            device='cuda',
        )

        for lg_edge in range(num_lg_edges):
            source = source_lg_nodes[lg_edge]
            destination = destination_lg_nodes[lg_edge]

            connecting_g_node = destination_g_nodes[source]

            for head in range(num_heads):
                attention_score = node_self_attention[head][connecting_g_node]

                attention_projection[head][source][destination] = attention_score

        return attention_projection

    def _create_edge_attention_projection(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        node_self_attention: torch.Tensor,
    ) -> torch.Tensor:

        edge_attention_projection = self._edge_attention_script(
            node_self_attention,
            lg.edges()[0],
            lg.edges()[1],
            g.edges()[1],
            self._num_heads,
            g.num_edges(),
            lg.num_edges(),
        )

        return edge_attention_projection

    def _calculate_message_passing(
        self,
        g: dgl.DGLGraph,
        key: torch.Tensor,
        attention_projection: torch.Tensor,
    ) -> torch.Tensor:
        if self._message_aggregation_type == 'sum':
            x = attention_projection
        elif self._message_aggregation_type == 'mean':
            degree_inv = torch.linalg.inv(torch.diag(g.in_degrees().float()))

            x = degree_inv @ attention_projection
        elif self._message_aggregation_type == 'gcn':
            adjacency = g.adj(ctx=self._device.device).to_dense()

            degree_inv_sqrt = torch.sqrt(torch.linalg.inv(
                torch.diag(g.in_degrees().float())))
            adjacency_inv_sqrt = torch.sqrt(torch.linalg.inv(adjacency))

            x = degree_inv_sqrt @ attention_projection * \
                adjacency_inv_sqrt @ degree_inv_sqrt

        message_passing = x @ key

        return message_passing

    def forward(
        self,
        g: dgl.DGLGraph,
        g_adj: torch.Tensor,
        lg: dgl.DGLGraph,
        lg_adj: torch.Tensor,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_query_key = g_adj @ node_inputs
        node_query_key = self._node_query_key_linear_1(node_query_key)
        node_query_key = self._node_query_key_linear_2(node_query_key)
        node_query_key = node_query_key.view(self._num_heads, -1, 1)
        node_value = self._node_value_linear_1(node_inputs)
        node_value = self._node_value_linear_2(node_value)
        node_value = node_value.view(self._num_heads, -1, self._node_in_feats)

        edge_query_key = lg_adj @ edge_inputs
        edge_query_key = self._edge_query_key_linear_1(edge_query_key)
        edge_query_key = self._edge_query_key_linear_2(edge_query_key)
        edge_query_key = edge_query_key.view(self._num_heads, -1, 1)
        edge_value = self._edge_value_linear_1(edge_inputs)
        edge_value = self._edge_value_linear_2(edge_value)
        edge_value = edge_value.view(self._num_heads, -1, self._edge_in_feats)

        node_self_attention = self._calculate_self_attention(
            node_query_key, self._node_in_feats)
        edge_self_attention = self._calculate_self_attention(
            edge_query_key, self._edge_in_feats)

        node_attention_projection = self._create_node_attention_projection(
            g, edge_self_attention)
        edge_attention_projection = self._create_edge_attention_projection(
            g, lg, node_self_attention)

        node_message_passing = self._calculate_message_passing(
            g, node_value, node_attention_projection)
        edge_message_passing = self._calculate_message_passing(
            lg, edge_value, edge_attention_projection)

        node_embedding = node_message_passing * node_value
        edge_embedding = edge_message_passing * edge_value

        node_embedding = self._node_dropout(node_embedding)
        edge_embedding = self._edge_dropout(edge_embedding)

        if self._head_pooling_type == 'sum':
            node_embedding = node_embedding.sum(dim=-3)
            edge_embedding = edge_embedding.sum(dim=-3)
        elif self._head_pooling_type == 'mean':
            node_embedding = node_embedding.mean(dim=-3)
            edge_embedding = edge_embedding.mean(dim=-3)

        return node_embedding, edge_embedding


class MutualAttentionTransformerLayer(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_out_feats: int,
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
        self._node_linear_embedding = LinearBlock(
            node_in_feats, 
            node_out_feats, 
            activation=linear_embedding_activation,
        )
        self._edge_linear_embedding = LinearBlock(
            edge_in_feats, 
            edge_out_feats, 
            activation=linear_embedding_activation,
        )

        if normalization_type == 'layer':
            self._node_normalization_1 = nn.LayerNorm(node_in_feats)
            self._node_normalization_2 = nn.LayerNorm(node_out_feats)

            self._edge_normalization_1 = nn.LayerNorm(edge_in_feats)
            self._edge_normalization_2 = nn.LayerNorm(edge_out_feats)
        elif normalization_type == 'batch':
            self._node_normalization_1 = nn.BatchNorm1d(node_in_feats)
            self._node_normalization_2 = nn.BatchNorm1d(node_out_feats)

            self._edge_normalization_1 = nn.BatchNorm1d(edge_in_feats)
            self._edge_normalization_2 = nn.BatchNorm1d(edge_out_feats)

    def forward(
        self,
        g: dgl.DGLGraph,
        g_adj: torch.Tensor,
        lg: dgl.DGLGraph,
        lg_adj: torch.Tensor,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embedding, edge_embedding = self._mutual_multi_attention_head(
            g, g_adj, lg, lg_adj, node_inputs, edge_inputs)

        if self._long_residual:
            node_embedding += node_inputs
            edge_embedding += edge_inputs

        node_embedding = self._node_normalization_1(node_embedding)
        edge_embedding = self._edge_normalization_1(edge_embedding)

        node_embedding = self._node_linear_embedding(node_embedding)
        edge_embedding = self._edge_linear_embedding(edge_embedding)

        node_embedding = self._node_normalization_2(node_embedding)
        edge_embedding = self._edge_normalization_2(edge_embedding)

        return node_embedding, edge_embedding


class GraphMutualAttentionTransformer(nn.Module):
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
        long_residual: bool,
        dropout_probability: float,
        message_aggregation_type: str,
        head_pooling_type: str,
        readout_pooling_type: str,
        normalization_type: str,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
        bilinear_readout_activation: str = None,
    ) -> None:
        super().__init__()
        self._node_out_feats = node_out_feats
        self._edge_out_feats = edge_out_feats
        self._num_layers = num_layers
        self._transformer_layers = self._create_transformer_layers(
            node_in_feats,
            node_hidden_feats,
            node_out_feats,
            edge_in_feats,
            edge_hidden_feats,
            edge_out_feats,
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
        self._node_linear_1 = LinearBlock(
            node_out_feats,
            node_out_feats * 16,
            normalization='batch',
            activation='leaky_relu',
        )
        self._node_linear_2 = LinearBlock(
            node_out_feats * 16,
            node_out_feats,
            normalization='batch',
            activation='leaky_relu',
        )
        self._edge_linear_1 = LinearBlock(
            edge_out_feats,
            edge_out_feats * 16,
            normalization='batch',
            activation='leaky_relu',
        )
        self._edge_linear_2 = LinearBlock(
            edge_out_feats * 16,
            edge_out_feats,
            normalization='batch',
            activation='leaky_relu',
        )

        self._bilinear_readout = BilinearBlock(
            node_out_feats, edge_out_feats, 1, bilinear_readout_activation)

        if readout_pooling_type == 'sum':
            self._readout_pooling = dgl.nn.pytorch.SumPooling()
        elif readout_pooling_type == 'mean':
            self._readout_pooling = dgl.nn.pytorch.AvgPooling()
        elif readout_pooling_type == 'attention':
            pass

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
        long_residual: bool,
        dropout_probability: float,
        message_aggregation_type: str,
        head_pooling_type: str,
        normalization_type: str,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
    ) -> nn.ModuleList:
        transformer_layers = nn.ModuleList()

        if num_layers > 1:
            transformer_layers.append(MutualAttentionTransformerLayer(
                node_in_feats,
                node_hidden_feats,
                edge_in_feats,
                edge_hidden_feats,
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
                transformer_layers.append(MutualAttentionTransformerLayer(
                    node_hidden_feats,
                    node_hidden_feats,
                    edge_hidden_feats,
                    edge_hidden_feats,
                    num_heads,
                    long_residual,
                    dropout_probability,
                    message_aggregation_type,
                    head_pooling_type,
                    normalization_type,
                    linear_projection_activation,
                    linear_embedding_activation,
                ))

            transformer_layers.append(MutualAttentionTransformerLayer(
                node_hidden_feats,
                node_out_feats,
                edge_hidden_feats,
                edge_out_feats,
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
            transformer_layers.append(MutualAttentionTransformerLayer(
                node_in_feats,
                node_out_feats,
                edge_in_feats,
                edge_out_feats,
                num_heads,
                long_residual,
                dropout_probability,
                message_aggregation_type,
                head_pooling_type,
                normalization_type,
                linear_projection_activation,
                linear_embedding_activation,
            ))

        return transformer_layers

    def forward(
        self,
        g: dgl.DGLGraph,
        g_adj: torch.Tensor,
        lg: dgl.DGLGraph,
        lg_adj: torch.Tensor,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> torch.Tensor:
        node_embedding = node_inputs
        edge_embedding = edge_inputs

        for transformer_layer in self._transformer_layers:
            node_embedding, edge_embedding = transformer_layer(
                g, g_adj, lg, lg_adj, node_embedding, edge_embedding)

        node_embedding = self._node_linear_1(node_embedding)
        node_embedding = self._node_linear_2(node_embedding)
        edge_embedding = self._edge_linear_1(edge_embedding)
        edge_embedding = self._edge_linear_2(edge_embedding)

        node_embedding = self._readout_pooling(g, node_embedding)
        edge_embedding = self._readout_pooling(lg, edge_embedding)

        readout = self._bilinear_readout(node_embedding, edge_embedding)

        return readout
