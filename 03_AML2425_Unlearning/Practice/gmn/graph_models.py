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
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # src, dest: [E, F_x], where E is the number of edges. src is the source node features and dest is the destination node features of each edge.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: only here it will have shape [E] with max entry B - 1, because here it indicates the graph index for each edge.

        '''
        Add your code below
        '''
        
        #E = total number of edges for every graph
        #B = number of graphs in each batch
        #batch is an array that indicates which edges belong to which graph using consecutive numbers
        #example [0, 0, 1, 1, 1, 1, 2, 3, 3] 2 edges for 1st graph, 4 edges for 2nd graph, 1 edge for 3rd graph, 2 edges for 4th graph
        
        #we have to concatenate src,dest,edge_attr,u
        #shape E x (2F_x + F_e + F_u)
        #but u has shape B x F_u so we have to expand it to E x F_u
        #we do that by replicating each row of u for each edge belonging to the same graph
        #example [u[0], u[0], u[1], u[1], u[1], u[1], u[2], u[3], u[3]]
        
        u_expanded = u[batch]
        x = torch.cat([dest, src, edge_attr, u_expanded], dim=-1)
        
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
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        '''
        Add your code below
        '''
        
        sources = x[edge_index[0]]
        dests = x[edge_index[1]]
        print(dests.shape)
        
        #global features for each target edge
        u_edge = u[batch[edge_index[1]]]        #E * F_u
        
        edge_input = torch.cat([dests, sources, edge_attr, u_edge, ], dim=-1)
        
        edge_attr = self.node_mlp_1(edge_input)
        
        # aggregated_out = scatter(edge_attr, edge_index[1], dim=0, reduce=self.reduce)
        aggregated_out = scatter(edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce=self.reduce)
        
        u_expanded = u[batch]
        
        new_input = torch.cat([x, aggregated_out, u_expanded], dim=-1)
        
        assert x.size(1) + aggregated_out.size(1) + u_expanded.size(1) == self.node_mlp_2[0].in_features, \
    f"Mismatch in input dimensions for node_mlp_2: got {x.size(1) + aggregated_out.size(1) + u_expanded.size(1)}, " \
    f"expected {self.node_mlp_2[0].in_features}"

        
        output = self.node_mlp_2(new_input)
        
        return output


class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, reduce='sum'):
        super().__init__()
        if activation:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr, u, batch):
        #**IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        '''
        Add your code below
        '''
        
        node_agg = scatter(x, batch, dim=0, reduce=self.reduce)  # [B, F_x]
        
        edge_agg = scatter(edge_attr, batch[edge_index[1]], dim=0, reduce=self.reduce)  # [B, F_e]

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

        assert num_layers >= 2
        
        # First layer (MetaLayer)
        edge_model = EdgeModel(in_dim=2*node_in_dim + edge_in_dim + global_in_dim, out_dim=hidden_dim)
        node_model = NodeModel(in_dim_mlp1=2*node_in_dim + hidden_dim + global_in_dim, 
                               in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, 
                               out_dim=hidden_dim)
        global_model = GlobalModel(in_dim=2*hidden_dim + global_in_dim, out_dim=hidden_dim)

        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
        self.node_norms.append(nn.BatchNorm1d(hidden_dim))
        self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
        self.global_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers-2):
            edge_model = EdgeModel(in_dim=4*hidden_dim, out_dim=hidden_dim)
            node_model = NodeModel(in_dim_mlp1=4*hidden_dim, 
                                in_dim_mlp2=3*hidden_dim, 
                                out_dim=hidden_dim)
            global_model = GlobalModel(in_dim=3*hidden_dim, out_dim=hidden_dim)

            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
            self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            self.global_norms.append(nn.BatchNorm1d(hidden_dim))

        edge_model = EdgeModel(in_dim=4*hidden_dim, out_dim=edge_out_dim, activation=False)
        node_model = NodeModel(in_dim_mlp1=3*hidden_dim + edge_out_dim, 
                                in_dim_mlp2=2*hidden_dim + edge_out_dim, 
                                out_dim=node_out_dim, activation=False)
        global_model = GlobalModel(in_dim=hidden_dim + edge_out_dim + node_out_dim, out_dim=global_out_dim, activation=False)   

        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))

    

    def forward(self, x, edge_index, edge_attr, u, batch, *args):
        for i, conv in enumerate(self.convs):
            
            # Apply MetaLayer
            x, edge_attr, u = conv(x, edge_index, edge_attr, u, batch)
            
            if i != len(self.convs)-1 and self.use_bn:
                x = self.node_norms[i](x)
                edge_attr = self.edge_norms[i](edge_attr)
                u = self.global_norms[i](u)

                # Dropout
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
                u = F.dropout(u, p=self.dropout, training=self.training)
        
        return x, edge_attr, u
