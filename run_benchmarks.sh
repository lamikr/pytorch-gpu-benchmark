#! /bin/bash

if [ -x "$(command -v rocm-smi)" ]; then
    count=`rocm-smi --showproductname --json | wc -l`

    echo 'AMD gpu benchmarks starting'
    echo "GPU count: " ${count}
    for (( c=$count; c>=1; c-- ))
    do
        python3 benchmark_models.py -g $c&& &>/dev/null
    done
    echo 'AMD GPU benchmarks finished'
fi

if [ -x "$(command -v nvidia-smi)" ]; then
    count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

    echo 'NVidia gpu benchmarks starting'
    echo "GPU count: " ${count}
    for (( c=$count; c>=1; c-- ))
    do
        python3 benchmark_models.py -g $c&& &>/dev/null
    done
    echo 'Nvidia GPU benchmarks finished'
fi
