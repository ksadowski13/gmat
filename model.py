from typing import Tuple

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.key_linear = nn.Linear(in_feats, 1)
        self.value_linear = nn.Linear(in_feats, in_feats)

    def forward(self, inputs):
        key = self.key_linear(inputs)
        value = self.value_linear(inputs)

        x = value @ inputs.t() @ key
        x = F.softmax(x, dim=0)

        return x


class MutualAttentionGraphConv(nn.Module):
    def __init__(self, in_feats, aggregation_function: str):
        super().__init__()
        self.aggregation_function = aggregation_function
        self.linear = nn.Linear(in_feats, in_feats)

    def forward(
        self,
        inputs: torch.Tensor(),
        attention_adjacency: torch.Tensor(),
    ):
        if self.aggregation_function == 'sum':
            x = attention_adjacency @ inputs
        elif self.aggregation_function == 'mean':
            degree = torch.diag(torch.count_nonzero(
                attention_adjacency, dim=1))
            degree_inverse = torch.linalg.inv(degree.float())

            x = degree_inverse @ attention_adjacency @ inputs

        x = self.linear(x)

        return x


class Head(nn.Module):
    def __init__(self, node_in_feats: int, edge_in_feats: int) -> None:
        super().__init__()
        self.node_attention = SelfAttention(node_in_feats)
        self.node_query_linear = nn.Linear(node_in_feats, node_in_feats)
        self.node_query_conv = MutualAttentionGraphConv(node_in_feats, 'mean')
        self.edge_attention = SelfAttention(edge_in_feats)
        self.edge_query_linear = nn.Linear(edge_in_feats, edge_in_feats)
        self.edge_query_conv = MutualAttentionGraphConv(edge_in_feats, 'sum')

    def create_mutual_node_attention_adjacency(
        self,
        g: dgl.DGLGraph,
        edge_attention: torch.Tensor,
    ) -> torch.Tensor:
        node_adjacency = g.adj().to_dense()

        for edge in range(g.num_edges()):
            nodes = g.find_edges(edge)

            source_node = nodes[0].item()
            destination_node = nodes[1].item()

            node_adjacency[source_node][destination_node] = edge_attention[edge]

        return node_adjacency

    def create_mutual_edge_attention_adjacency(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        node_attention: torch.Tensor,
    ) -> torch.Tensor:
        edge_adjacency = lg.adj().to_dense()

        for node in range(lg.num_edges()):
            edges = lg.find_edges(node)

            source_edge = edges[0].item()
            destination_edge = edges[1].item()

            connecting_node = g.find_edges(source_edge)[1]

            edge_adjacency[source_edge][destination_edge] = node_attention[connecting_node]

        return edge_adjacency

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_attention = self.node_attention(node_feats)
        edge_attention = self.edge_attention(edge_feats)

        node_mutual_attention_adjacency = self.create_mutual_node_attention_adjacency(
            g, edge_attention)
        node_query = self.node_query_linear(node_feats)
        node_query = self.node_query_conv(
            node_query, node_mutual_attention_adjacency)

        edge_mutual_attention_adjacency = self.create_mutual_edge_attention_adjacency(
            g, lg, node_attention)
        edge_query = self.edge_query_linear(edge_feats)
        edge_query = self.edge_query_conv(
            edge_query, edge_mutual_attention_adjacency)

        return node_query, edge_query
