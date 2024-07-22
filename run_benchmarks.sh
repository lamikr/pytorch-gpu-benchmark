#! /bin/bash

count=$(python pytorch_device_count.py)

echo "GPU benchmarks starting"
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
echo "GPU benchmarks finished"
