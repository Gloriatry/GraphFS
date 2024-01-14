#!/bin/bash

WORKSPACE="/glusterfs/dgs_test"
DATASET="cora"
NUM_PARTS=2
PART_METHOD="metis"
XFEAT_NAME="xfeat_mask"
XFEAT_RATIO=0.99
OUTPUT="${WORKSPACE}/data/${XFEAT_RATIO}.${DATASET}.${PART_METHOD}.${NUM_PARTS}"

python3 ${WORKSPACE}/data/partition_graph.py \
    --dataset ${DATASET} \
    --num_parts ${NUM_PARTS} \
    --part_method ${PART_METHOD} \
    --balance_train \
    --undirected \
    --balance_edges \
    --num_trainers_per_machine 1 \
    --output ${OUTPUT} \
    --xfeat_name ${XFEAT_NAME} \
    --xfeat_ratio ${XFEAT_RATIO}
