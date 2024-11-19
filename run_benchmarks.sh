#! /bin/bash

count=$(python pytorch_device_count.py)

echo "Pytorch-GPU-Benchmark started"
echo "GPU count: " ${count}
for (( ii = 0; ii < $count; ii++ ))
do
	# run benchmark for one gpu at a time
	echo "Pytorch-GPU-Benchmark, testing GPU: $ii"
	python3 benchmark_models.py -i $ii -g 1&& &>/dev/null
	echo "Pytorch-GPU-Benchmark test ready, GPU: $ii"
done
if (( count > 1 )); then
	# if there is more than 1 gpu, run benchmark again by not specifying the gpu
	echo "Multi-GPU benchmark, GPU count: $count"
	#python3 benchmark_models.py -g $count&& &>/dev/null
fi
echo "Pytorch-GPU-Benchmark done"
