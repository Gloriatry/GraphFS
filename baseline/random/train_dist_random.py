import os

os.environ['DGLBACKEND'] = 'pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import random
from sklearn.metrics import f1_score

from os import path
import sys

test_src = path.abspath(path.join(path.dirname(__file__)))
src_dir = path.abspath(path.join(test_src, os.pardir))
sys.path.append(src_dir)
sys.path.append("/nfsroot/dgs")

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data.utils import load_graphs
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.distributed import DistDataLoader
from scipy import sparse
from tensorboardX import SummaryWriter

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from models.sage import DistSAGE, NeighborSampler
from models.gcn import GCN
from models.gat import GAT
from utils import *
"""
Generate lap_matrix, use random_sampler for sampling
"""

multilabel_data = set(['amazon'])
XFEAT_RATIO = 0.8
EVAL_EVERY = 5


def init_seeds(args):
    if args.seed == 0:
        return
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    # Can be fast when NN structure and input are fixed
    torch.backends.cudnn.benchmark = True


def load_subtensor(g, seeds, input_nodes, device, idx_select=None):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    if idx_select is None:
        batch_inputs = g.ndata['features'][input_nodes].to(device)
        batch_labels = g.ndata['labels'][seeds].to(device)
    else:
        # batch_inputs = g.ndata['features'].index_select(
        #     input_nodes, 1, idx_select).to(device)
        batch_inputs = g.ndata['features'][input_nodes].index_select(1, idx_select).to(device)
        batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels


# class NeighborSampler(object):
#     def __init__(self,
#                  g,
#                  fanouts,
#                  sample_neighbors,
#                  prob,
#                  lap_matrix,
#                  device,
#                  load_feat=True):
#         self.g = g
#         self.fanouts = fanouts
#         self.sample_neighbors = sample_neighbors

#         self.device = device
#         self.load_feat = load_feat

#         self.feat_size = self.g.ndata['features'].shape[1]

#     def sample_blocks(self, seeds):
#         seeds = th.LongTensor(np.asarray(seeds))
#         blocks = []
#         for fanout in self.fanouts:
#             # For each seed node, sample ``fanout`` neighbors.
#             frontier = self.sample_neighbors(self.g,
#                                              seeds,
#                                              fanout,
#                                              replace=True)
#             # Then we compact the frontier into a bipartite graph for message passing.
#             block = dgl.to_block(frontier, seeds)
#             # Obtain the seed nodes for next layer.
#             seeds = block.srcdata[dgl.NID]

#             blocks.insert(0, block)

#         input_nodes = blocks[0].srcdata[dgl.NID]
#         seeds = blocks[-1].dstdata[dgl.NID]

#         batch_inputs = th.zeros([len(input_nodes), self.feat_size])

#         node_feat_mask = th.from_numpy(
#             np.random.choice(self.feat_size,
#                              int(self.feat_size * XFEAT_RATIO),
#                              replace=True))

#         batch_in, batch_labels = load_subtensor(self.g,
#                                                 seeds,
#                                                 input_nodes,
#                                                 "cpu",
#                                                 idx_select=node_feat_mask)

#         batch_inputs[:, node_feat_mask] = batch_in

#         blocks[0].srcdata['features'] = batch_inputs
#         blocks[-1].dstdata['labels'] = batch_labels
#         return blocks


def compute_acc(pred, labels, multilabel):
    """
    Compute the accuracy of prediction given the labels.
    """
    if not multilabel:
        labels = labels.long()
        acc = (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        pred[pred > 0] = 1
        pred[pred <= 0] = 0
        acc = f1_score(labels.detach().cpu(),
                       pred.detach().cpu(),
                       average="micro")
    return acc


def evaluate(model,
             g,
             inputs,
             labels,
             val_nid,
             test_nid,
             batch_size,
             prob,
             lap_matrix,
             device,
             multilabel=False,
             heads=None):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    if heads is None:
        with th.no_grad():
            # pred = model.inference(g, inputs, NeighborSampler, batch_size,
            #                        prob, lap_matrix, device)
            pred = model.inference(g, inputs, batch_size, device)
    else:
        with th.no_grad():
            # pred = model.inference(g, inputs, NeighborSampler, batch_size,
            #                        prob, lap_matrix, device, heads)
            pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid],
                       multilabel), compute_acc(pred[test_nid],
                                                labels[test_nid], multilabel)


def pad_data(nids, device):
    """
    In distributed traning scenario, we need to make sure that each worker has same number of
    batches. Otherwise the synchronization(barrier) is called diffirent times, which results in
    the worker with more batches hangs up.

    This function pads the nids to the same size for all workers, by repeating the head ids till
    the maximum size among all workers.
    """
    import torch.distributed as dist
    num_nodes = th.tensor(nids.numel()).to(device)
    dist.all_reduce(num_nodes, dist.ReduceOp.MAX)
    max_num_nodes = int(num_nodes)
    nids_length = nids.shape[0]
    if max_num_nodes > nids_length:
        pad_size = max_num_nodes % nids_length
        repeat_size = max_num_nodes // nids_length
        new_nids = th.cat([nids
                           for _ in range(repeat_size)] + [nids[:pad_size]],
                          axis=0)
        print("Pad nids from {} to {}".format(nids_length, max_num_nodes))
    else:
        new_nids = nids
    assert new_nids.shape[0] == max_num_nodes
    return new_nids


def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, n_classes, multilabel, g = data
    # train_nid = pad_data(train_nid, device)
    # adjacency matrix: for fastgcn
    # adj_matrix = g.local_partition.adjacency_matrix_scipy(
    #     return_edge_ids=False).astype(float)
    adj_matrix = g.local_partition.adj(scipy_fmt='csr')
    lap_matrix = row_normalize(adj_matrix + sparse.eye(adj_matrix.shape[0]))

    # Create sampler
    sampler = NeighborSampler(
        g, [int(fanout) for fanout in args.fan_out.split(',')[1:]],
        dgl.distributed.sample_neighbors, device)

    # Create DataLoader for constructing blocks
    dataloader = DistDataLoader(dataset=train_nid.numpy(),
                                batch_size=args.batch_size,
                                collate_fn=sampler.sample_blocks,
                                shuffle=True,
                                drop_last=False)

    # 测试用的dataloader
    dataloader_test = DistDataLoader(dataset=val_nid.numpy(),
                                batch_size=args.batch_size,
                                collate_fn=sampler.sample_blocks,
                                shuffle=True,
                                drop_last=False)

    # Define model and optimizer
    if args.model.lower() == 'gcn':
        model = GCN(in_feats, args.num_hidden, n_classes, args.num_layers,
                    F.relu, args.dropout)
    elif args.model.lower() == 'sage':
        model = DistSAGE(in_feats, args.num_hidden, n_classes, args.num_layers,
                         F.relu, args.dropout)
    elif args.model.lower() == 'gat':
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
        model = GAT(args.num_layers, in_feats, args.num_hidden, n_classes,
                    heads, F.relu, args.in_drop, args.attn_drop,
                    args.negative_slope, args.residual)

    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id)
    if multilabel:
        loss_fcn = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_size = th.sum(g.ndata['train_mask'][0:g.number_of_nodes()])

    writer = SummaryWriter(f'logs/{args.graph_name}_rank{g.rank()}_logs')

    # Training loop
    iter_tput = []
    epoch = 0
    total_step = 0
    eval_times = 0  # 测试的次数
    for epoch in range(args.num_epochs):
        #log_nic(args, epoch)
        tic = time.time()

        sample_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()
            sample_time += tic_step - start

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']
            if not multilabel:
                batch_labels = batch_labels.long()

            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])
            blocks = [block.to(device) for block in blocks]
            batch_labels = batch_labels.to(device)

            # Compute loss and prediction
            fp_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            forward_end = time.time()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - fp_start
            backward_time += compute_end - forward_end

            optimizer.step()
            update_time += time.time() - compute_end

            step_t = time.time() - start
            step_time.append(step_t)
            iter_tput.append(len(blocks[-1].dstdata[dgl.NID]) / step_t)
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels, multilabel)
                gpu_mem_alloc = th.cuda.max_memory_allocated(
                ) / 1000000 if th.cuda.is_available() else 0
                print(
                    'Part {} | Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB | time {:.3f} s'
                    .format(g.rank(), epoch, step, loss.item(), acc.item(),
                            np.mean(iter_tput[3:]), gpu_mem_alloc,
                            np.sum(step_time[-args.log_every:])))
                writer.add_scalar('Train Accuracy', acc.item(), total_step)
            if total_step % EVAL_EVERY == 0:
                if eval_times % dataloader_test.expected_idxs == 0:
                    loader = iter(dataloader_test)
                model.eval()
                with th.no_grad():
                    blocks = next(loader)
                    batch_inputs = blocks[0].srcdata['features']
                    batch_labels = blocks[-1].dstdata['labels']
                    if not multilabel:
                        batch_labels = batch_labels.long()
                    blocks = [block.to(device) for block in blocks]
                    batch_labels = batch_labels.to(device)
                    batch_pred = model(blocks, batch_inputs)
                    eval_acc = compute_acc(batch_pred, batch_labels, multilabel)
                    print('Epoch {:05d} | Step {:05d} | Val Acc {:.4f}'.format(epoch, step, eval_acc))
                    writer.add_scalar('Test Accuracy', eval_acc.item(), total_step)
                model.train()
                eval_times += 1
            total_step += 1
            start = time.time()

        toc = time.time()
        print(
            'Part {}, Epoch Time(s): {:.4f}, sample+data_copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'
            .format(g.rank(), toc - tic, sample_time, forward_time,
                    backward_time, update_time, num_seeds, num_inputs))
        epoch += 1

        # 测试的时候也用minibatch测试，而非在整个图上测试，很慢
        # if epoch % args.eval_every == 0 and epoch != 0:
        #     start = time.time()
        #     if args.model.lower() == 'gat':
        #         val_acc, test_acc = evaluate(model.module,
        #                                      g,
        #                                      g.ndata['features'],
        #                                      g.ndata['labels'],
        #                                      val_nid,
        #                                      test_nid,
        #                                      args.batch_size_eval,
        #                                      args.prob,
        #                                      lap_matrix,
        #                                      device,
        #                                      multilabel=multilabel,
        #                                      heads=heads)
        #     else:
        #         val_acc, test_acc = evaluate(model.module,
        #                                      g,
        #                                      g.ndata['features'],
        #                                      g.ndata['labels'],
        #                                      val_nid,
        #                                      test_nid,
        #                                      args.batch_size_eval,
        #                                      args.prob,
        #                                      lap_matrix,
        #                                      device,
        #                                      multilabel=multilabel)

        #     print('Part {}, Val Acc {:.4f}, Test Acc {:.4f}, time: {:.4f}'.
        #           format(g.rank(), val_acc, test_acc,
        #                  time.time() - start))
        #     writer.add_scalar('Test Accuracy',test_acc, epoch)

    #log_nic(args, epoch)
    writer.close()


def main(args):
    multilabel = args.graph_name in multilabel_data

    init_seeds(args)
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend=args.backend)
    g = dgl.distributed.DistGraph(args.graph_name,
                                  part_config=args.part_config)
    print('rank:', g.rank())

    pb = g.get_partition_book()
    if 'trainer_id' in g.ndata:
        train_nid = dgl.distributed.node_split(
            g.ndata['train_mask'],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata['trainer_id'])
        val_nid = dgl.distributed.node_split(
            g.ndata['val_mask'],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata['trainer_id'])
        test_nid = dgl.distributed.node_split(
            g.ndata['test_mask'],
            pb,
            force_even=True,
            node_trainer_ids=g.ndata['trainer_id'])
    else:
        train_nid = dgl.distributed.node_split(g.ndata['train_mask'],
                                               pb,
                                               force_even=True)
        val_nid = dgl.distributed.node_split(g.ndata['val_mask'],
                                             pb,
                                             force_even=True)
        test_nid = dgl.distributed.node_split(g.ndata['test_mask'],
                                              pb,
                                              force_even=True)
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    print(
        'part {}, train: {} (local: {}), val: {} (local: {}), test: {} (local: {})'
        .format(g.rank(), len(train_nid),
                len(np.intersect1d(train_nid.numpy(),
                                   local_nid)), len(val_nid),
                len(np.intersect1d(val_nid.numpy(), local_nid)), len(test_nid),
                len(np.intersect1d(test_nid.numpy(), local_nid))))
    if args.num_gpus == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:' + str(g.rank() % args.num_gpus))

    if multilabel:
        n_classes = g.ndata['labels'].shape[1]
    else:
        labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
        n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    print('#labels:', n_classes)

    # Pack data
    in_feats = g.ndata['features'].shape[1]
    data = train_nid, val_nid, test_nid, in_feats, n_classes, multilabel, g
    run(args, device, data)
    print("parent ends")


def log_nic(args, epoch):
    import errno
    fname = 'nic.{}.{}/{}.{}.{}-parts'.format("random", args.graph_name,
                                              args.id, args.model,
                                              args.num_clients)

    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    f = open(fname, "a+")
    f.write("Epoch: {}\n".format(epoch))

    # cmd = "./nic.sh {} {}".format(fname, "0")
    # os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--seed', type=int, help='init seed')
    parser.add_argument('--ip_config',
                        type=str,
                        help='The file for IP configuration')
    parser.add_argument('--part_config',
                        type=str,
                        help='The path to the partition config file')
    parser.add_argument('--num_clients',
                        type=int,
                        help='The number of clients')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_hidden', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)

    # GAT
    parser.add_argument("--num-heads",
                        type=int,
                        default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads",
                        type=int,
                        default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual",
                        action="store_true",
                        default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop",
                        type=float,
                        default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop",
                        type=float,
                        default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope',
                        type=float,
                        default=0.2,
                        help="the negative slope of leaky relu")

    parser.add_argument('--local_rank',
                        type=int,
                        help='get rank of the process')
    parser.add_argument('--standalone',
                        action='store_true',
                        help='run in the standalone mode')
    parser.add_argument('--prob',
                        type=str,
                        help='probability of fastgcn sampling')

    parser.add_argument('--feat_size', type=int, help='Size of feature size')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='pytorch distributed backend')
    args = parser.parse_args()

    print(args)
    main(args)
