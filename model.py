from typing import List, Tuple

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
            degree_inv_sqrt = torch.sqrt(torch.linalg.inv(
                torch.diag(g.in_degrees().float())))
            adjacency_inv_sqrt = torch.sqrt(torch.linalg.inv(adjacency))

            x = degree_inv_sqrt @ attention_projection * \
                adjacency_inv_sqrt @ degree_inv_sqrt @ inputs

        x = self._linear(x)

        return x


class MutualMultiAttentionHead(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_out_feats: int,
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
        self._node_query_linear = LinearLayer(
            node_in_feats, num_heads * node_in_feats, linear_projection_activation)
        self._node_key_linear = LinearLayer(
            node_in_feats, num_heads, linear_projection_activation)
        self._node_value_linear = LinearLayer(
            node_in_feats, num_heads * node_in_feats, linear_projection_activation)
        self._edge_query_linear = LinearLayer(
            edge_in_feats, num_heads * edge_in_feats, linear_projection_activation)
        self._edge_key_linear = LinearLayer(
            edge_in_feats, num_heads, linear_projection_activation)
        self._edge_value_linear = LinearLayer(
            edge_in_feats, num_heads * edge_in_feats, linear_projection_activation)
        self._node_convolution = GraphAttentionConvolution(
            node_in_feats, node_out_feats, convolution_aggregation_type, convolution_activation)
        self._edge_convolution = GraphAttentionConvolution(
            edge_in_feats, edge_out_feats, convolution_aggregation_type, convolution_activation)

    def _calculate_self_attention(self, query: torch.Tensor, key: torch.Tensor, short_residual: torch.Tensor = None) -> torch.Tensor:
        if short_residual is not None:
            x = query @ short_residual.T @ key
        else:
            x = query @ torch.transpose(query, -1, -2) @ key

        x = F.softmax(x, dim=1)

        return x

    def _create_node_attention_projection(self, g: dgl.DGLGraph, edge_self_attention: torch.Tensor) -> torch.Tensor:
        node_attention_projection = torch.zeros(
            [self._num_heads, g.num_nodes(), g.num_nodes()])

        for edge in range(g.num_edges()):
            nodes = g.find_edges(edge)

            source = nodes[0].item()
            destination = nodes[1].item()

            for head in range(self._num_heads):
                attention_score = edge_self_attention[edge][head]

                node_attention_projection[head][source][destination] = attention_score

            return node_attention_projection

    def _create_edge_attention_projection(self, g: dgl.DGLGraph, lg: dgl.DGLGraph, node_self_attention: torch.Tensor) -> torch.Tensor:
        edge_attention_projection = torch.zeros(
            [self._num_heads, g.num_edges(), g.num_edges()])

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
            node_self_attention = self._calculate_self_attention(
                node_query, node_key, node_inputs)
            edge_self_attention = self._calculate_self_attention(
                edge_query, edge_key, edge_inputs)
        else:
            node_self_attention = self._calculate_self_attention(
                node_query, node_key)
            edge_self_attention = self._calculate_self_attention(
                edge_query, edge_key)

        node_attention_projection = self._create_node_attention_projection(
            g, edge_self_attention)
        edge_attention_projection = self._create_edge_attention_projection(
            g, lg, node_self_attention)

        node_outputs = self._node_convolution(
            g, node_value, node_attention_projection)
        edge_outputs = self._edge_convolution(
            lg, edge_value, edge_attention_projection)

        return node_outputs, edge_outputs


class MutualAttentionTransformerLayer(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_out_feats: int,
        num_heads: int,
        short_residual_connection: bool,
        long_residual_connection: bool,
        convolution_aggregation_type: str,
        embedding_aggregation_type: str,
        normalization_type: str,
        convolution_activation: str = None,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
    ) -> None:
        super().__init__()
        self._long_residual_connection = long_residual_connection
        self._embedding_aggregation_type = embedding_aggregation_type
        self._normalization_type = normalization_type
        self._mutual_multi_attention_head = MutualMultiAttentionHead(
            node_in_feats,
            node_out_feats,
            edge_in_feats,
            edge_out_feats,
            num_heads,
            short_residual_connection,
            convolution_aggregation_type,
            convolution_activation,
            linear_projection_activation,
        )
        self._node_linear_embedding = LinearLayer(
            node_out_feats, node_out_feats, linear_embedding_activation)
        self._edge_linear_embedding = LinearLayer(
            edge_out_feats, edge_out_feats, linear_embedding_activation)

        if normalization_type == 'layer':
            self._node_normalization = nn.LayerNorm(node_out_feats)
            self._edge_normalization = nn.LayerNorm(edge_out_feats)
        elif normalization_type == 'batch':
            self._node_normalization = nn.BatchNorm1d(node_out_feats)
            self._edge_normalization = nn.BatchNorm1d(edge_out_feats)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        node_inputs: torch.Tensor,
        edge_inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embedding, edge_embedding = self._mutual_multi_attention_head(
            g, lg, node_inputs, edge_inputs)

        node_embedding = self._node_linear_embedding(node_embedding)
        edge_embedding = self._edge_linear_embedding(edge_embedding)

        if self._embedding_aggregation_type == 'sum':
            node_embedding = node_embedding.sum(dim=0)
            edge_embedding = edge_embedding.sum(dim=0)
        elif self._embedding_aggregation_type == 'mean':
            node_embedding = node_embedding.mean(dim=0)
            edge_embedding = edge_embedding.mean(dim=0)

        if self._long_residual_connection:
            node_embedding += node_inputs
            edge_embedding += edge_inputs

        node_embedding = self._node_normalization(node_embedding)
        edge_embedding = self._edge_normalization(edge_embedding)

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
        num_transfomer_layers: int,
        num_heads: int,
        short_residual_connection: bool,
        long_residual_connection: bool,
        convolution_aggregation_type: str,
        embedding_aggregation_type: str,
        readout_aggregation_type: str,  # 
        normalization_type: str,
        convolution_activation: str = None,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
    ) -> None:
        self._transfomer_layers = self._create_transformer_layers(
            node_in_feats,
            node_hidden_feats,
            node_out_feats,
            edge_in_feats,
            edge_hidden_feats,
            edge_out_feats,
            num_transfomer_layers,
            num_heads,
            short_residual_connection,
            long_residual_connection,
            convolution_aggregation_type,
            embedding_aggregation_type,
            normalization_type,
            convolution_activation,
            linear_projection_activation,
            linear_embedding_activation,
        )

    def _create_transformer_layers(
        self,
        node_in_feats: int,
        node_hidden_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_hidden_feats: int,
        edge_out_feats: int,
        num_transfomer_layers: int,
        num_heads: int,
        short_residual_connection: bool,
        long_residual_connection: bool,
        convolution_aggregation_type: str,
        embedding_aggregation_type: str,
        normalization_type: str,
        convolution_activation: str = None,
        linear_projection_activation: str = None,
        linear_embedding_activation: str = None,
    ) -> List[nn.ModuleList]:
        transformer_layers = nn.ModuleList()

        if num_transfomer_layers > 1:
            transformer_layers.append(MutualAttentionTransformerLayer(
                node_in_feats,
                node_hidden_feats,
                edge_in_feats,
                edge_hidden_feats,
                num_heads,
                short_residual_connection,
                long_residual_connection,
                convolution_aggregation_type,
                embedding_aggregation_type,
                normalization_type,
                convolution_activation,
                linear_projection_activation,
                linear_embedding_activation,
            ))

            for _ in range(num_transfomer_layers - 2):
                transformer_layers.append(MutualAttentionTransformerLayer(
                    node_hidden_feats,
                    node_hidden_feats,
                    edge_hidden_feats,
                    edge_hidden_feats,
                    num_heads,
                    short_residual_connection,
                    long_residual_connection,
                    convolution_aggregation_type,
                    embedding_aggregation_type,
                    normalization_type,
                    convolution_activation,
                    linear_projection_activation,
                    linear_embedding_activation,
                ))

            transformer_layers.append(MutualAttentionTransformerLayer(
                node_hidden_feats,
                node_out_feats,
                edge_hidden_feats,
                edge_out_feats,
                num_heads,
                short_residual_connection,
                long_residual_connection,
                convolution_aggregation_type,
                embedding_aggregation_type,
                normalization_type,
                convolution_activation,
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
                short_residual_connection,
                long_residual_connection,
                convolution_aggregation_type,
                embedding_aggregation_type,
                normalization_type,
                convolution_activation,
                linear_projection_activation,
                linear_embedding_activation,
            ))

        return transformer_layers
