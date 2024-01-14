import dgl
import numpy as np
import torch as th
import argparse
import time
import os
from os import path
import sys
from pathlib import Path
import pdb

from load_graph_with_prob import load_karate, load_cora, load_reddit, load_ogb, create_simple_graph


def dataset_split(s):
    s = s.strip()
    h = s.split('.')
    return h


def partition(args, cluster_dict, partid2rank_dict=None):
    m = dataset_split(args.dataset)
    print(m)
    args.dataset = m[0]
    if len(m) == 4:
        # by default, g3 partition
        n = int(m[3])
    elif len(m) == 5:
        # other partition
        n = int(m[3])
        p = m[4]

    if len(m) > 1:
        assert n == len(cluster_dict), 'partition != cluster scale'

    start = time.time()
    if args.dataset == 'simple':
        g = create_simple_graph(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'karate':
        g, _ = load_karate(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'cora':
        g, _ = load_cora(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'reddit':
        # g, _ = load_reddit(prob=args.prob, feat_size=args.feat_size)
        g = load_reddit(prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogb-paper100M':
        g, _ = load_ogb('ogbn-papers100M',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogb-proteins':
        g, _ = load_ogb('ogbn-proteins',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogb-mag':
        g, _ = load_ogb('ogbn-mag', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogb-arxiv':
        g, _ = load_ogb('ogbn-arxiv', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogbn-products':
        g, _ = load_ogb('ogbn-products',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogbn-papers100M':
        g, _ = load_ogb('ogbn-papers100M',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogbn-proteins':
        g, _ = load_ogb('ogbn-proteins',
                        prob=args.prob,
                        feat_size=args.feat_size)
    elif args.dataset == 'ogbn-mag':
        g, _ = load_ogb('ogbn-mag', prob=args.prob, feat_size=args.feat_size)
    elif args.dataset == 'ogbn-arxiv':
        g, _ = load_ogb('ogbn-arxiv', prob=args.prob, feat_size=args.feat_size)

    print('load {} takes {:.3f} seconds'.format(args.dataset,
                                                time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}, test: {}'.format(
        th.sum(g.ndata['train_mask']), th.sum(g.ndata['val_mask']),
        th.sum(g.ndata['test_mask'])))
    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    if args.gat:
        # add self loop
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        output = args.output + '.gat'
    else:
        output = args.output

    if args.part_method == "chunk":
        f = open("{}/{}.{}.{}.{}".format(args.manual_dir,
                                         args.dataset, "manual",
                                         len(cluster_dict), args.part_method))
        loaded_parts = [int(x) for x in f.readlines()]

        dgl.distributed.partition_graph(g,
                                        args.dataset,
                                        len(cluster_dict),
                                        output,
                                        num_hops=args.num_layer,
                                        part_method="manual",
                                        reshuffle=args.reshuffle,
                                        balance_ntypes=balance_ntypes,
                                        balance_edges=args.balance_edges,
                                        node_ps=loaded_parts)
    else:
        print("Start partitioning...")
        # pdb.set_trace()
        dgl.distributed.partition_graph(g,
                                        args.dataset,
                                        len(cluster_dict),
                                        output,
                                        num_hops=args.num_layer,
                                        part_method=args.part_method,
                                        # reshuffle=args.reshuffle,
                                        balance_ntypes=balance_ntypes,
                                        balance_edges=args.balance_edges)


def InitClusterDict(clusterfile):
    cluster_dict = {}
    partid2rank_dict = {}
    f = open(clusterfile, 'r')
    lines = f.readlines()

    for i, line in enumerate(lines):
        # ip, _, port, th_port, partid, rank = line.split()
        ip = line.split()
        cluster_dict[i] = ip

    return cluster_dict


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        '--dataset',
        type=str,
        default='cora',
        help='datasets: simple, cora, reddit, ogb-product, ogb-paper100M')
    argparser.add_argument('--manual_dir',
                           type=str,
                           default=None,
                           help='manual_dir')
    argparser.add_argument('--interface',
                           type=str,
                           default='eth0',
                           help='network interface')
    argparser.add_argument('--username',
                           type=str,
                           default='xinchen',
                           help='ssh username')
    argparser.add_argument("--clusterfile",
                           type=str,
                           default="nodes.txt",
                           help="Distributed cluster settings")
    argparser.add_argument('--part_method',
                           type=str,
                           default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train',
                           action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected',
                           action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument(
        '--balance_edges',
        action='store_true',
        help='balance the number of edges in each partition.')
    argparser.add_argument(
        '--reshuffle',
        action='store_true',
        help='balance the number of edges in each partition.')
    argparser.add_argument('--output',
                           type=str,
                           default='../data',
                           help='Output path of partitioned graph.')
    argparser.add_argument(
        '--sync-dir',
        type=str,
        default='../data',
        help='Sync absolute path of partitioned graph to remote machine.')
    argparser.add_argument('--gat',
                           action='store_true',
                           help='used for gat model?')
    argparser.add_argument('--prob', type=str, help='name of probability')
    argparser.add_argument('--num_layer', type=int, help='number of layers')
    argparser.add_argument('--feat_size', type=int, help='size of features')
    argparser.add_argument('--partition',
                           action='store_true',
                           help='process partition function')
    argparser.add_argument('--sync',
                           action='store_true',
                           help='process sync function')
    argparser.add_argument('--sync-all',
                           action='store_true',
                           help='process sync function')
    args = argparser.parse_args()

    print(args)
    cluster_dict = InitClusterDict(args.clusterfile)

    # partition
    if args.partition:
        partition(args, cluster_dict)
