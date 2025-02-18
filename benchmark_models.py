"""Compare speed of different models with batch size 12"""
import torch
import torchvision.models as models
import platform
import psutil
import torch.nn as nn
import datetime
import time
import os
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import json
import sys
from dataclasses import dataclass

@dataclass
class BenchmarkModelData:
    model_desc: str
    model_set: dict

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.

PRECISION_LIST_SMALL = ["float"]
PRECISION_LIST_MEDIUM = ["float", "double"]
PRECISION_LIST_LARGE = ["half", "float", "double"]

# mnasnet0_5,mnasnet0_75
# resnet18,resnet34,resnet50
# densenet121,
# squeezenet1_0,
# vgg11,vgg11_bn,
# mobilenet_v3_small,
# shufflenet_v2_x0_5,shufflenet_v2_x1_0, shufflenet_v2_x1_5,shufflenet_v2_x2_0
MODEL_LIST_SMALL = {
    models.mnasnet: ["mnasnet0_5", "mnasnet0_75"],
    models.resnet: ["resnet18", "resnet34", "resnet50", "resnet101"],
    models.densenet: ["densenet121", "densenet161"],
    models.squeezenet: ["squeezenet1_0", "squeezenet1_1"],
    models.vgg: ["vgg11", "vgg11_bn"],
    models.mobilenet: ["mobilenet_v3_large", "mobilenet_v3_small"],
}

# mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3
# resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d,wide_resnet50_2,wide_resnet101_2,
# densenet121,densenet161,densenet169,densenet201
# squeezenet1_0,squeezenet1_1,
# vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,
# mobilenet_v3_large,mobilenet_v3_small,
# shufflenet_v2_x0_5,shufflenet_v2_x1_0, shufflenet_v2_x1_5,shufflenet_v2_x2_0
MODEL_LIST_MEDIUM = {
    models.mnasnet: ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0"],
    models.resnet: ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    models.densenet: ["densenet121", "densenet161"],
    models.squeezenet: ["squeezenet1_0", "squeezenet1_1"],
    models.vgg: ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn"],
    models.mobilenet: ["mobilenet_v3_large", "mobilenet_v3_small"],
    models.shufflenetv2: ["shufflenet_v2_x0_5", "shufflenet_v2_x1_5"],
}

# mnasnet0_5,mnasnet0_75,mnasnet1_0,mnasnet1_3
# resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,resnext101_64x4d,wide_resnet50_2,wide_resnet101_2,
# densenet121,densenet161,densenet169,densenet201
# squeezenet1_0,squeezenet1_1,
# vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16,vgg16_bn,vgg19,vgg19_bn,
# mobilenet_v3_large,mobilenet_v3_small,
# shufflenet_v2_x0_5,shufflenet_v2_x1_0, shufflenet_v2_x1_5,shufflenet_v2_x2_0
MODEL_LIST_LARGE = {
    models.mnasnet: models.mnasnet.__all__[1:],
    models.resnet: models.resnet.__all__[1:],
    models.densenet: models.densenet.__all__[1:],
    models.squeezenet: models.squeezenet.__all__[1:],
    models.vgg: models.vgg.__all__[1:],
    models.mobilenet: models.mobilenet.mv3_all[1:],
    models.shufflenetv2: models.shufflenetv2.__all__[1:],
}

# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Benchmarking")
parser.add_argument(
    "--WARM_UP", "-w", type=int, default=5, required=False, help="Num of warm up"
)      

parser.add_argument(
    "--NUM_TEST", "-n", type=int, default=50, required=False, help="Num of Test"
)

parser.add_argument(
    "--BATCH_SIZE", "-b", type=int, default=4, required=False, help="Num of batch size"
)

parser.add_argument(
    "--NUM_CLASSES", "-c", type=int, default=200, required=False, help="Num of class"
)

parser.add_argument(
    "--GPU_COUNT", "-g", type=int, default=1, required=False, help="Number of gpus used in test"
)

parser.add_argument(
    "--GPU_INDEX", "-i", type=int, default=-1, required=False, help="Index for the used gpu"
)

parser.add_argument(
    "--folder",
    "-f",
    type=str,
    default="new_results",
    required=False,
    help="folder to save results",
)

class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.randn(3, 224, 224, length)

    def __getitem__(self, index):
        return self.data[:, :, :, index]

    def __len__(self):
        return self.len

def train(cur_precision="single", gpu_index=-1, benchmark_model=MODEL_LIST_SMALL):
    """use fake image for training speed test"""
    if gpu_index >= 0:
        target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda(gpu_index)
    else:
        target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in benchmark_model.keys():
        for model_name in benchmark_model[model_type]:
            if model_name[-8:] == '_Weights': continue
            torch_device_name = "cuda"
            if (gpu_index >= 0):
                torch_device_name = "cuda:" + str(gpu_index)
                torch.cuda.set_device(gpu_index)
            #print("torch_device_name: " + torch_device_name)
            model = getattr(model_type, model_name)()
            if args.GPU_COUNT > 1:
                model = nn.DataParallel(model, device_ids=range(args.GPU_COUNT))
            model = getattr(model, cur_precision)()
            torch_device = torch.device(torch_device_name)
            model = model.to(torch_device)
            durations = []
            print(f"Benchmarking training, precision: {cur_precision}, model: {model_name}")
            for step, img in enumerate(rand_loader):
                img = getattr(img, cur_precision)()
                torch.cuda.synchronize()
                start = time.time()
                model.zero_grad()
                prediction = model(img.to("cuda"))
                loss = criterion(prediction, target)
                loss.backward()
                torch.cuda.synchronize()
                end = time.time()
                if step >= args.WARM_UP:
                    durations.append((end - start) * 1000)
            print(f"Average training time: {sum(durations)/len(durations)} ms, model: {model_name}")
            del model
            torch.cuda.empty_cache()
            benchmark[model_name] = durations
            #print(torch.cuda.memory_summary())
    return benchmark

def inference(cur_precision="float", gpu_index=-1, benchmark_model=MODEL_LIST_SMALL):
    benchmark = {}
    with torch.no_grad():
        for model_type in benchmark_model.keys():
            for model_name in benchmark_model[model_type]:
                if model_name[-8:] == '_Weights': continue
                torch_device_name = "cuda"
                if (gpu_index >= 0):
                    torch_device_name = "cuda:" + str(gpu_index)
                    torch.cuda.set_device(gpu_index)
                #print("torch_device_name: " + torch_device_name)
                torch_device = torch.device(torch_device_name)
                model = getattr(model_type, model_name)()
                if args.GPU_COUNT > 1:
                    model = nn.DataParallel(model, device_ids=range(args.GPU_COUNT))
                model = getattr(model, cur_precision)()
                model = model.to(torch_device)
                model.eval()
                durations = []
                print(f"Benchmarking inference, precision: {cur_precision}, model: {model_name}")
                for step, img in enumerate(rand_loader):
                    img = getattr(img, cur_precision)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to(torch_device))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start) * 1000)
                print(f"Average inference time: {sum(durations)/len(durations)} ms, model: {model_name}")
                del model
                torch.cuda.empty_cache()
                benchmark[model_name] = durations
                #print(torch.cuda.memory_summary())
    return benchmark

if __name__ == "__main__":
    args = parser.parse_args()
    args.BATCH_SIZE *= args.GPU_COUNT
    gpu_count = args.GPU_COUNT
    gpu_index = args.GPU_INDEX

    print("GPU_ID: " + str(gpu_index) + "/" + str(gpu_count))
    rand_loader = DataLoader(
        dataset=RandomDataset(args.BATCH_SIZE * (args.WARM_UP + args.NUM_TEST)),
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )

    # get available memory
    gpu_mem_total = 0
    gpu_mem_used = 0
    if (gpu_index >= 0):
        device_name = str(torch.cuda.get_device_name(gpu_index))
        mem_tuple = torch.cuda.mem_get_info(gpu_index)
        gpu_mem_total = mem_tuple[1]
        gpu_mem_free = mem_tuple[0]
    else:
        device_name = str(torch.cuda.get_device_name(0))
        # search which gpu has smallest amount of memory
        for ii in range (gpu_count):
            mem_tuple = torch.cuda.mem_get_info(ii)
            if (ii == 0):
                gpu_mem_total = mem_tuple[0]
                gpu_mem_free = mem_tuple[1]
            else:
                if (mem_tuple[0] < gpu_mem_total):
                    gpu_mem_total = mem_tuple[0]
                    gpu_mem_free = mem_tuple[1]
    # convert memory sizes to gigabytes (gb)
    gpu_mem_total = gpu_mem_total / (1048576.0 * 1024)
    gpu_mem_free = gpu_mem_free / (1048576.0 * 1024)
    gpu_mem_used = gpu_mem_total - gpu_mem_free

    # select which set of benchmarks and precisions to run
    # depending from the gpu memory available. (to avoid out of memory errors)
    benchmark_model_dict = {}
    # if gpu is AMD's integrated graphic card, run only the minime set of benchmarks
    if device_name == "AMD Radeon Graphics":
        precision_list_arr = PRECISION_LIST_LARGE
        model_list_arr = MODEL_LIST_SMALL
    else:
        if (gpu_mem_free <= 6):
            precision_list_arr = PRECISION_LIST_MEDIUM
            model_list_arr = MODEL_LIST_MEDIUM
        elif (gpu_mem_free > 6) and (gpu_mem_free <= 10):
            precision_list_arr = PRECISION_LIST_LARGE
            model_list_arr = MODEL_LIST_MEDIUM
        else:
            precision_list_arr = PRECISION_LIST_LARGE
            model_list_arr = MODEL_LIST_LARGE
    if (precision_list_arr == PRECISION_LIST_SMALL):
        precision_list_name = "SMALL"
    elif (precision_list_arr == PRECISION_LIST_MEDIUM):
        precision_list_name = "MEDIUM"
    else:
        precision_list_name = "LARGE"
    if (model_list_arr == MODEL_LIST_SMALL):
        model_list_name = "SMALL"
    elif (model_list_arr == MODEL_LIST_MEDIUM):
        model_list_name = "MEDIUM"
    else:
        model_list_name = "LARGE"

    for ii, cur_prec in enumerate(precision_list_arr):
        benchmark_model_dict[cur_prec] = BenchmarkModelData(model_list_name, model_list_arr)

    device_name = f"{device_name}"
    if (args.GPU_COUNT > 1):
        device_name = device_name + str(gpu_count) + "X"
    device_name = device_name.replace(" ", "_")
    device_file_name = device_name + "_"
    print("GPU Name: " + device_name)
    print("GPU Memory: " + str(gpu_mem_total) + " GB")
    print("GPU Memory Used: " + str(gpu_mem_used) + " GB")
    print("GPU Memory Free: " + str(gpu_mem_free) + " GB")

    if (gpu_index >= 0):
        folder_name = args.folder + "/" + str(gpu_index) + "/" + device_name
    else:
        folder_name = args.folder + "/" + str(gpu_count) + "X"
    print("Result directory: " + folder_name)
    system_info_list = f"\
	            OS: {platform.uname()}\n\
				CPU Freq: {psutil.cpu_freq()}\n\
				CPU Count: {psutil.cpu_count()}\n\
				System Memory: {psutil.virtual_memory().available}\n"
    gpu_config_header_list = [
        "GPU_Count: ",
        "Torch_Version : ",
        "ROCM_Version: ",
        "CUDA_Version: ",
        "Cudnn_Version: ",
        "Device_Name: ",
        "GPU Mem Total (GB): ",
        "GPU Mem Used (GB): ",
        "GPU Mem Free (GB): ",
    ]
    gpu_config_info_list = [
        gpu_count,
        torch.__version__,
        torch.version.hip,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        device_name,
        gpu_mem_total,
        gpu_mem_used,
        gpu_mem_free,
    ]
    gpu_config_info_list = list(map(str, gpu_config_info_list))

    model_config_header_list = [
        "Benchmark Model List Name: ",
        "Benchmark Precision List Name: ",
        "Benchmark Precision List: ",
        "Batch Size: ",
    ]
    model_config_info_list = [
        model_list_name,
        precision_list_name,
        precision_list_arr,
        str(args.BATCH_SIZE),
    ]
    model_config_info_list = list(map(str, model_config_info_list))

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    datetime_now = datetime.datetime.now()
    start_time_str = datetime_now.strftime("%Y/%m/%d %H:%M:%S")

    print(system_info_list)

    for idx, value in enumerate(zip(gpu_config_header_list, gpu_config_info_list)):
        gpu_config_info_list[idx] = "".join(value)
        print(gpu_config_info_list[idx])

    for idx, value in enumerate(zip(model_config_header_list, model_config_info_list)):
        model_config_info_list[idx] = "".join(value)
        print(model_config_info_list[idx])

    with open(os.path.join(folder_name, "system_info.txt"), "w") as f:
        f.writelines(f"1) Benchmark start time:\n{start_time_str}\n")
        f.writelines("1) System configuration:\n\n")
        f.writelines(system_info_list)
        f.writelines("2) GPU configuration:\n\n")
        f.writelines(s + "\n" for s in gpu_config_info_list)
        f.writelines("3) Model configuration:\n\n")
        f.writelines(s + "\n" for s in model_config_info_list)

    print("\nModels")
    indx=0
    for model_type in model_list_arr.keys():
        #print("train model_type: " + model_type)
        for model_name in model_list_arr[model_type]:
            indx=indx+1
            print("  model[" + str(indx) + "]: " + model_name)

    print(f"\nBenchmark start time: {start_time_str}")
    for cur_precision in precision_list_arr:
        benchmark_model_data = benchmark_model_dict[cur_precision]
        print("Precision: " + cur_precision + ", model list: " + benchmark_model_data.model_desc)

        train_result = train(cur_precision, gpu_index, benchmark_model_data.model_set)
        train_result_df = pd.DataFrame(train_result)
        path = f"{folder_name}/{device_file_name}_{cur_precision}_model_train_benchmark.csv"
        train_result_df.to_csv(path, index=False)

        inference_result = inference(cur_precision, gpu_index, benchmark_model_data.model_set)
        inference_result_df = pd.DataFrame(inference_result)
        path = f"{folder_name}/{device_file_name}_{cur_precision}_model_inference_benchmark.csv"
        inference_result_df.to_csv(path, index=False)

    # finish the benchmarks
    datetime_now = datetime.datetime.now()
    end_time_str = datetime_now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"Benchmark end time: {end_time_str}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"Benchmark end time: {end_time_str}\n")
