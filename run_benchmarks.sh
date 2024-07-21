#! /bin/bash

if [ -x "$(command -v rocm-smi)" ]; then
    count=`amd-smi list | grep UUID: | wc -l`

    echo 'AMD gpu benchmarks starting'
    echo "GPU count: " ${count}
    for (( ii = 0; ii < $count; ii++ ))
    do
        # run benchmark for one gpu at a time
		echo "benchmark gpu index: $ii"
        python3 benchmark_models.py -i $ii -g 1&& &>/dev/null
    done
    if (( count > 1 )); then
        # then if there are more than 1 gpu, run benchmark which allows using all of them
        echo "multigpu benchmark: $count"
        python3 benchmark_models.py -g $count&& &>/dev/null
    fi
    echo 'AMD GPU benchmarks finished'
fi

if [ -x "$(command -v nvidia-smi)" ]; then
    count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

    echo 'NVidia gpu benchmarks starting'
    echo "GPU count: " ${count}
    for (( ii = 0; ii < $count; ii++ ))
    do
        # run benchmark for one gpu at a time
        python3 benchmark_models.py -i $ii -g 1&& &>/dev/null
        echo $ii
    done
    if (( count > 1 )); then
        # then if there are more than 1 gpu, run benchmark which allows using all of them
        echo "multigpu benchmark: $count"
        python3 benchmark_models.py -g $count&& &>/dev/null
    fi
    echo 'Nvidia GPU benchmarks finished'
fi
