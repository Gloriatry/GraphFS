#!/bin/bash

WORKSPACE="/home/yp/dgs"
IP_CONFIG=$1
DATASET=$2

NUM_PARTS=$(wc -l ../../${IP_CONFIG} | awk '{print $1}')

INTERFACE="0"
USER="yp"
PART_METHOD=$3
PROB="prob"
NUM_LAYER=2

FEAT_SIZE=$4
if [[ ${FEAT_SIZE} -eq -1 ]]; then
    OUTPUT="${WORKSPACE}/dataset/${DATASET}.${PART_METHOD}.${NUM_PARTS}.prob"
    SYNCDIR="${WORKSPACE}/dataset/${DATASET}.${PART_METHOD}.${NUM_PARTS}.prob"
else
    OUTPUT="${WORKSPACE}/dataset/${DATASET}.${PART_METHOD}.${NUM_PARTS}.f${FEAT_SIZE}.prob"
    SYNCDIR="${WORKSPACE}/dataset/${DATASET}.${PART_METHOD}.${NUM_PARTS}.f${FEAT_SIZE}.prob"
fi

MANUALDIR="${WORKSPACE}/baseline/partition"

cd ${WORKSPACE} && python3 baseline/partition/partition_graph.py \
    --dataset ${DATASET} \
    --manual_dir ${MANUALDIR} \
    --interface ${INTERFACE} \
    --username ${USER} \
    --clusterfile ${IP_CONFIG} \
    --part_method ${PART_METHOD} \
    --balance_train \
    --balance_edges \
    --reshuffle \
    --output ${OUTPUT} \
    --prob ${PROB} \
    --num_layer ${NUM_LAYER} \
    --feat_size ${FEAT_SIZE} \
    --partition
