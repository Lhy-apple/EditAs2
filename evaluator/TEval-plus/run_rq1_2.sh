#!/bin/bash

N_JOBS=${1:-10}
START=1
END=1
g=toga
MODELNAME=ModelName
for i in `seq ${START} ${END}`;do
    echo ${i}
    for t in buggy;do
        echo ${t}
        echo ${g}
        bash run_exp.sh data/evosuite_${t}_tests/${i} data/evosuite_${t}_regression_all/${i} ${N_JOBS} ${g} ${MODELNAME}
    done
    rm -rf /tmp/run_bug_detection.pl_*
done
