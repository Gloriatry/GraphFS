import numpy as np
import tqdm

import dgl
import dgl.nn.pytorch as dglnn
from dgl.distributed import DistDataLoader

import torch as th
import torch.nn as nn

from . import context
from sampling import DistGNNSampler
from explainer import dummy_layer
from helper.logger import logger
from helper import DGS_ERROR
from .layers import FeatureSelector

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors, device):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.device = device

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=False)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)

        input_nodes = blocks[0].srcdata[dgl.NID]
        output_nodes = blocks[-1].dstdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]
        blocks[0].srcdata['features'] = self.g.ndata['features'][input_nodes]
        blocks[-1].dstdata['labels'] = self.g.ndata['labels'][output_nodes]
        # print('sample one block time is {}'.format(s_t))

        return blocks


# class DistSAGE(nn.Module):
#     def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
#                  dropout):
#         super().__init__()
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.n_classes = n_classes
#         self.layers = nn.ModuleList()
#         self.layers.append(
#             dummy_layer(dglnn.SAGEConv, in_feats, n_hidden, 'mean'))
#         for i in range(1, n_layers - 1):
#             self.layers.append(
#                 dummy_layer(dglnn.SAGEConv, n_hidden, n_hidden, 'mean'))
#         self.layers.append(
#             dummy_layer(dglnn.SAGEConv, n_hidden, n_classes, 'mean'))

#         self.dropout = nn.Dropout(dropout)
#         self.activation = activation

#     def forward(self, blocks, x, e_weights=None, is_explain=False):
#         h = x
#         for l, (layer, block) in enumerate(zip(self.layers, blocks)):
#             h = layer(block, h, e_weights, is_explain)
#             if l != len(self.layers) - 1:
#                 h = self.activation(h)
#                 h = self.dropout(h)
#         return h

#     def inference(self, xeprod_name, g, x, batch_size, device):
#         """
#         Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
#         g : the entire graph.
#         x : the input of entire node set.
#         The inference code is written in a fashion that it could handle any number of nodes and
#         layers.
#         """
#         # During inference with sampling, multi-layer blocks are very inefficient because
#         # lots of computations in the first few layers are repeated.
#         # Therefore, we compute the representation of all nodes layer by layer.  The nodes
#         # on each layer are of course splitted in batches.
#         # TODO: can we standardize this?
#         nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()),
#                                            g.get_partition_book(),
#                                            force_even=True)
#         y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_hidden),
#                                        th.float32,
#                                        'h',
#                                        persistent=True)
#         for l, layer in enumerate(self.layers):
#             if l == len(self.layers) - 1:
#                 y = dgl.distributed.DistTensor(
#                     (g.number_of_nodes(), self.n_classes),
#                     th.float32,
#                     'h_last',
#                     persistent=True)

#             # Use default 'features'
#             sampler = DistGNNSampler(xeprod_name, g, [-1],
#                                      dgl.distributed.sample_neighbors, device)
#             print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(),
#                                                        batch_size))
#             # Create PyTorch DataLoader for constructing blocks
#             dataloader = DistDataLoader(dataset=nodes,
#                                         batch_size=batch_size,
#                                         collate_fn=sampler.sample_blocks,
#                                         shuffle=False,
#                                         drop_last=False)

#             for blocks in tqdm.tqdm(dataloader):
#                 block = blocks[0].to(device)
#                 input_nodes = block.srcdata[dgl.NID]
#                 output_nodes = block.dstdata[dgl.NID]
#                 h = x[input_nodes].to(device)
#                 h_dst = h[:block.number_of_dst_nodes()]
#                 h = layer(block, (h, h_dst))
#                 if l != len(self.layers) - 1:
#                     h = self.activation(h)
#                     h = self.dropout(h)

#                 y[output_nodes] = h.cpu()

#             x = y
#             g.barrier()
#         return y


class DistSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()),
                                           g.get_partition_book(), force_even=True)
        y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_hidden), th.float32, 'h',
                                       persistent=True)
        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_classes),
                                               th.float32, 'h_last', persistent=True)

            sampler = NeighborSampler(g, [-1], dgl.distributed.sample_neighbors, device)
            print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(), batch_size))
            # Create PyTorch DataLoader for constructing blocks
            dataloader = DistDataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False)

            for blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

class STGDistSAGE(DistSAGE):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, device, sigma=1.0, lam=0.1):
        super().__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
        self.FeatureSelector = FeatureSelector(in_feats, sigma, device)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma
        
    def forward(self, blocks, x):
        x = self.FeatureSelector(x)
        logits = super().forward(blocks, x)
        return logits