import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation, normalization


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        normalization: str = None,
        activation: str = None,
        dim: int = None,
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
            elif activation == 'softmax' and dim is not None:
                self._activation = nn.Softmax(dim=dim)
        else:
            self._activation = None

    def forward(self, inputs: torch.Tensor):
        x = self._linear(inputs)

        if self._normalization is not None:
            x = self._normalization(x)

        if self._activation is not None:
            x = self._activation(x)

        return x


class MessageProjection(nn.Module):
    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_heads: int,
        message_weight_func: str,
        reduce_func: str,
        weight_activation: str = None,
        message_activation: str = None,
    ):
        super().__init__()
        self._node_in_feats = node_in_feats
        self._edge_in_feats = edge_in_feats
        self._num_heads = num_heads

        if message_weight_func == 'add':
            self._message_weight_func = fn.u_add_e('feat', 'weight', 'message')
        elif message_weight_func == 'sub':
            self._message_weight_func = fn.u_sub_e('feat', 'weight', 'message')
        elif message_weight_func == 'mul':
            self._message_weight_func = fn.u_mul_e('feat', 'weight', 'message')
        elif message_weight_func == 'div':
            self._message_weight_func = fn.u_div_e('feat', 'weight', 'message')

        if reduce_func == 'sum':
            self._reduce_func = fn.sum('message', 'projection')
        elif reduce_func == 'mean':
            self._reduce_func = fn.mean('message', 'projection')

        self._edge_linear = LinearBlock(
            edge_in_feats, 
            1, 
            activation=weight_activation,
            dim=0 if weight_activation == 'softmax' else None,
        )
        self._message_linear = LinearBlock(
            node_in_feats, 
            node_in_feats * num_heads,
            activation=message_activation,
        )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        g.edata['weight'] = self._edge_linear(g.edata['feat'])

        g.update_all(
            message_func=self._message_weight_func,
            reduce_func=self._reduce_func,
        )

        message_projection = self._message_linear(g.ndata.pop('projection'))
        message_projection = message_projection.view(
            -1, self._num_heads, self._node_in_feats)

        edge_projection = g.edata['feat'] * g.edata.pop('weight')

        return message_projection, edge_projection

class LinearProjection(nn.Module):
    def __init__(
        self,
        in_feats: int,
        num_heads: int,
        activation: str = None,
    ):
        super().__init__()
        pass
