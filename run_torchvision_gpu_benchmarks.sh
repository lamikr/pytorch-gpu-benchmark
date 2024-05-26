#! /bin/bash
if [ -x "$(command -v rocm-smi)" ]; then
count=`rocm-smi --showproductname --json | wc -l`
	echo "start, count: " ${count}
	for (( curindx=${count}; curindx>=1; curindx-- ))
	do
		python3 benchmark_models_torchvision_013.py -g $curindx
	done
	echo 'end'
else
	echo "rocm-smi not installed"
fi
if [ -x "$(command -v nvidia-smi)" ]; then
count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
	echo 'start'
	for (( curindx=$count; curindx>=1; curindx-- ))
	do
		python3 benchmark_models_torchvision_013.py -g $curindx&& &>/dev/null
	done
	echo 'end'
else
	echo "nvidia-smi not installed"
fi
