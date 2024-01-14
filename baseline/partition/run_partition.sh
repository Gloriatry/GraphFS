#!/bin/bash

declare -a ip_config=("2nodes.txt")
declare -a dataset=("reddit")
declare -a part_method=("metis")
declare -a feat_size=(-1)

for ip in "${ip_config[@]}"; do
    for data in "${dataset[@]}"; do
        for part in "${part_method[@]}"; do
            for feat in "${feat_size[@]}"; do
                echo "./batch_partition.sh $ip $data $part $feat"
                ./batch_partition.sh $ip $data $part $feat
            done
        done
    done
done
