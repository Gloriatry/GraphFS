#!/bin/bash

DIR="baseline"

WORKSPACE="/nfsroot/gnnfs"
IP_CONFIG="ip_config.txt"
DATASET="reddit"
# DATASETPATH="data"
BASELINE="random"
# DATASET="ogbn-papers100M"
DATASETPATH="dataset"

NUM_PARTS=$(wc -l ${IP_CONFIG} | awk '{print $1}')
PART_METHOD="metis"
PART_CONFIG="${DATASETPATH}/${DATASET}.${PART_METHOD}.${NUM_PARTS}.prob/${DATASET}.json"

MODEL="sage"
BATCH_SIZE=1024
BATCH_SIZE_EVAL=1024
FAN_OUT=\",64,64,64,64\"
N_LAYERS=4
NUM_HIDDEN=256  #default 16

INTERFACE="enp5s0f1"
N_EPOCHS=5
LR=0.003
DROPOUT=0.0
PROB="prob"

SUFFIX="sampling"

LOGDIR="${DIR}/${BASELINE}/${DATASET}.${PART_METHOD}.${NUM_PARTS}.log.${N_LAYERS}layers.${MODEL}.${SUFFIX}"

cd ${WORKSPACE} && /home/yp/.conda/envs/dgs/bin/python launch.py \
    --workspace ${WORKSPACE} \
    --num_omp_threads 4 \
    --num_trainers 1 \
    --num_samplers 0 \
    --num_servers 1 \
    --part_config ${PART_CONFIG} \
    --ip_config ${IP_CONFIG} \
    --log_dir ${LOGDIR} \
    " NCCL_DEBUG=WARN \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=${INTERFACE} \
    PYTHONFAULTHANDLER=1 \
    /home/yp/.conda/envs/dgs/bin/python baseline/${BASELINE}/train_dist_random.py \
    --graph_name ${DATASET} \
    --dataset ${DATASET} \
    --seed 3 \
    --ip_config ${IP_CONFIG} \
    --part_config ${PART_CONFIG} \
    --num_clients ${NUM_PARTS} \
    --num_gpus 1 \
    --num_epochs ${N_EPOCHS} \
    --model ${MODEL} \
    --num_hidden ${NUM_HIDDEN} \
    --num_layers ${N_LAYERS} \
    --fan_out ${FAN_OUT} \
    --batch_size ${BATCH_SIZE} \
    --batch_size_eval ${BATCH_SIZE_EVAL} \
    --log_every 1 \
    --eval_every 1 \
    --lr ${LR} \
    --dropout ${DROPOUT} \
    --prob ${PROB} \
    --num-heads 8 \
    --num-out-heads 1 \
    --in-drop 0.6 \
    --attn-drop 0.6 \
    --negative-slope 0.2 "
