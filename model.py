from typing import Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, activation: str = None) -> None:
        super().__init__()
        self._linear = nn.Linear(in_feats, out_feats)
        self._activation = activation

    def forward(self, inputs: torch.Tensor) -> None:
        x = self._linear(inputs)
        
        if self._activation == 'relu':
            x = F.relu(x)

        return x

class GraphAttentionConvolution(nn.Module):
    def __init__(
        self, 
        in_feats: int,
        out_feats: int, 
        aggregation_type: str, 
        activation: str = None,
    ) -> torch.Tensor:
        super().__init__()
        self._aggregation_type = aggregation_type
        self._activation = activation
        self._linear = LinearLayer(in_feats, out_feats, activation)

    def forward(
        self, 
        g: dgl.DGLGraph, 
        inputs: torch.Tensor, 
        attention_projection: torch.Tensor,
    ) -> torch.Tensor:
        adjacency = g.adj().to_dense()

        if self._aggregation_type == 'sum':
            x = attention_projection * adjacency @ inputs
        elif self._aggregation_type == 'mean':
            degree_inv = torch.linalg.inv(torch.diag(g.in_degrees().float()))

            x = degree_inv @ attention_projection * adjacency @ inputs
        elif self._aggregation_type == 'gcn':
            degree_inv_sqrt = torch.sqrt(torch.linalg.inv(torch.diag(g.in_degrees().float())))
            adjacency_inv_sqrt = torch.sqrt(torch.linalg.inv(adjacency))

            x = degree_inv_sqrt @ attention_projection * adjacency_inv_sqrt @ degree_inv_sqrt @ inputs

        x = self._linear(x)

        return x


class MutualMultiAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_heads: int,
        short_residual_connection: bool,
        convolution_aggregation_type: str,
        convolution_activation: str = None,
        linear_projection_activation: str = None,
    ) -> None:
        super().__init__()
        self._node_in_feats = node_in_feats
        self._edge_in_feats = edge_in_feats
        self._num_heads = num_heads
        self._short_residual_connection = short_residual_connection
        self._node_query_linear = LinearLayer(node_in_feats, num_heads * node_in_feats, linear_projection_activation)
        self._node_key_linear = LinearLayer(node_in_feats, num_heads, linear_projection_activation)
        self._node_value_linear = LinearLayer(node_in_feats, num_heads * node_in_feats, linear_projection_activation)
        self._edge_query_linear = LinearLayer(edge_in_feats, num_heads * edge_in_feats, linear_projection_activation)
        self._edge_key_linear = LinearLayer(edge_in_feats, num_heads, linear_projection_activation)
        self._edge_value_linear = LinearLayer(edge_in_feats, num_heads * edge_in_feats, linear_projection_activation)
        self._node_convolution = GraphAttentionConvolution(node_in_feats, node_in_feats, convolution_aggregation_type, convolution_activation)
        self._edge_convolution = GraphAttentionConvolution(edge_in_feats, edge_in_feats, convolution_aggregation_type, convolution_activation)

    def _calculate_self_attention(self, query: torch.Tensor, key: torch.Tensor, short_residual: torch.Tensor = None) -> torch.Tensor:
        if short_residual is not None:
            x = query @ short_residual.T @ key
        else:
            x = query @ torch.transpose(query, -1, -2) @ key

        x = F.softmax(x, dim=1)

        return x

    def _create_node_attention_projection(self, g: dgl.DGLGraph, edge_self_attention: torch.Tensor) -> torch.Tensor:
        node_attention_projection = torch.zeros([self._num_heads, g.num_nodes(), g.num_nodes()])

        for edge in range(g.num_edges()):
            nodes = g.find_edges(edge)

            source = nodes[0].item()
            destination = nodes[1].item()

            for head in range(self._num_heads):
                attention_score = edge_self_attention[edge][head]

                node_attention_projection[head][source][destination] = attention_score

            return node_attention_projection

    def _create_edge_attention_projection(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, node_self_attention: torch.Tensor) -> torch.Tensor:
        edge_attention_projection = torch.zeros([self._num_heads, g.num_edges(), g.num_edges()])

        for node in range(lg.num_edges()):
            edges = lg.find_edges(node)

            source = edges[0].item()
            destination = edges[1].item()

            connecting_node = g.find_edges(source)[1].item()

            for head in range(self._num_heads):
                attention_score = node_self_attention[head][connecting_node]

                edge_attention_projection[head][source][destination] = attention_score

            return edge_attention_projection

    def forward(
        self, 
        g: dgl.DGLGraph, 
        lg: dgl.DGLGraph, 
        node_inputs: torch.Tensor, 
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_query = self._node_query_linear(node_inputs)
        node_query = node_query.view(self._num_heads, -1, self._node_in_feats)
        node_key = self._node_key_linear(node_inputs)
        node_key = node_key.view(self._num_heads, -1, 1)
        node_value = self._node_value_linear(node_inputs)
        node_value = node_value.view(self._num_heads, -1, self._node_in_feats)

        edge_query = self._edge_query_linear(edge_inputs)
        edge_query = edge_query.view(self._num_heads, -1, self._edge_in_feats)
        edge_key = self._edge_key_linear(edge_inputs)
        edge_key = edge_key.view(self._num_heads, -1, 1)
        edge_value = self._edge_value_linear(edge_inputs)
        edge_value = edge_value.view(self._num_heads, -1, self._edge_in_feats)

        if self._short_residual_connection:
            node_self_attention = self._calculate_self_attention(node_query, node_key, node_inputs)
            edge_self_attention = self._calculate_self_attention(edge_query, edge_key, edge_inputs)
        else:
            node_self_attention = self._calculate_self_attention(node_query, node_key)
            edge_self_attention = self._calculate_self_attention(edge_query, edge_key)

        node_attention_projection = self._create_node_attention_projection(g, edge_self_attention)
        edge_attention_projection = self._create_edge_attention_projection(g, lg, node_self_attention)

        node_outputs = self._node_convolution(g, node_value, node_attention_projection).sum(dim=0)
        edge_outputs = self._edge_convolution(lg, edge_value, edge_attention_projection).sum(dim=0)

        return node_outputs, edge_outputs