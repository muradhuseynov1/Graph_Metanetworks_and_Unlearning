# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.pool import max_pool_x, avg_pool_x, global_max_pool, global_mean_pool
from torch_scatter import scatter

class EdgeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        # replace this with the class EdgeModel implemented by you in the theory part
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        # replace this with the forward function of the EdgeModel class implemented in the theory part
        u_expanded = u[batch]
        x = torch.cat((src, dest, edge_attr, u_expanded), dim=1)
        
        new_edge_attr = self.edge_mlp(x)
        
        
        return new_edge_attr


class NodeModel(nn.Module):
    def __init__(self, in_dim_mlp1, in_dim_mlp2, out_dim, activation=True, reduce='sum'):
        super().__init__()
        self.reduce = reduce
        if activation:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim), nn.ReLU())
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim), nn.ReLU())
        else:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim))
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim))
            
            
    def forward(self, x, edge_index, edge_attr, u, batch):
        # replace this with the forward function of the NodeModel class implemented in the theory part
        
        sources = x[edge_index[0]]
        dests = x[edge_index[1]]
        
        #global features for each source edge
        u_edge = u[batch[edge_index[0]]]
        
        edge_input = torch.cat([sources, dests, edge_attr, u_edge], dim=-1)
        
        edge_attr = self.node_mlp_1(edge_input)
        
        aggregated_out = scatter(edge_attr, edge_index[1], dim=0, reduce=self.reduce)
        
        u_expanded = u[batch]
        
        new_input = torch.cat((x, aggregated_out, u_expanded), dim=1)
        
        output = self.node_mlp_2(new_input)
        
        return output


class GlobalModel(nn.Module):
    # def __init__(self, in_dim, out_dim, reduce='sum'):
    def __init__(self, in_dim, out_dim, activation=True, reduce='sum'):
        super().__init__()
        # replace this with the class GlobalModel implemented by you in the theory part
        if activation:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr, u, batch):
        # replace this with the forward function of the GlobalModel class implemented in the theory part
        
        node_agg = scatter(x, batch, dim=0, reduce=self.reduce)  # [B, F_x]
        
        edge_agg = scatter(edge_attr, batch[edge_index[0]], dim=0, reduce=self.reduce)  # [B, F_e]

        input = torch.cat([node_agg, edge_agg, u], dim=-1)
        
        output = self.global_mlp(input)
        
        
        return output


class MPNN(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim, node_out_dim, edge_out_dim, global_out_dim, num_layers,
                use_bn=True, dropout=0.0, reduce='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.reduce = reduce
        
       #rest of the code here

    def forward(self, x, edge_index, edge_attr, u, batch, *args):

        for i, conv in enumerate(self.convs):
            '''
            Add your code below
            '''

            print(f"After layer {i}: x shape: {x.shape}, edge_attr shape: {edge_attr.shape}, u shape: {u.shape}")

            if i != len(self.convs)-1 and self.use_bn:
                '''
                Add your code below this line, but before the dropout
                '''
                x = self.node_norms[i](x)
                edge_attr = self.edge_norms[i](edge_attr)
                u = self.global_norms[i](u)


                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
                u = F.dropout(u, p=self.dropout, training=self.training)

        return x, edge_attr, u