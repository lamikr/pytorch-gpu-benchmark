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

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.

MODEL_LIST = {
    models.mnasnet: models.mnasnet.__all__[1:],
    models.resnet: models.resnet.__all__[1:],
    models.densenet: models.densenet.__all__[1:],
    models.squeezenet: models.squeezenet.__all__[1:],
    models.vgg: models.vgg.__all__[1:],
    models.mobilenet: models.mobilenet.mv2_all[1:],
    models.mobilenet: models.mobilenet.mv3_all[1:],
    models.shufflenetv2: models.shufflenetv2.__all__[1:],
}

precisions = ["float", "half", "double"]
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
    "--BATCH_SIZE", "-b", type=int, default=12, required=False, help="Num of batch size"
)

parser.add_argument(
    "--NUM_CLASSES", "-c", type=int, default=1000, required=False, help="Num of class"
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

def train(precision="single", gpu_index=-1):
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_type in MODEL_LIST.keys():
        for model_name in MODEL_LIST[model_type]:
            if model_name[-8:] == '_Weights': continue
            model = getattr(model_type, model_name)()
            if args.GPU_COUNT > 1:
                model = nn.DataParallel(model, device_ids=range(args.GPU_COUNT))
            model = getattr(model, precision)()
            torch_device_name = "cuda"
            if (gpu_index >= 0):
                torch_device_name = "cuda:" + str(gpu_index)
            print("torch_device_name: " + torch_device_name)
            torch_device = torch.device(torch_device_name)
            model = model.to(torch_device)        
            durations = []
            print(f"Benchmarking Training {precision} precision type {model_name} ")
            for step, img in enumerate(rand_loader):
                img = getattr(img, precision)()
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
            print(
                f"{model_name} model average train time : {sum(durations)/len(durations)}ms"
            )
            del model
            benchmark[model_name] = durations
    return benchmark

def inference(precision="float", gpu_index=-1):
    benchmark = {}
    with torch.no_grad():
        for model_type in MODEL_LIST.keys():
            for model_name in MODEL_LIST[model_type]:
                if model_name[-8:] == '_Weights': continue
                model = getattr(model_type, model_name)()
                if args.GPU_COUNT > 1:
                    model = nn.DataParallel(model, device_ids=range(args.GPU_COUNT))
                model = getattr(model, precision)()
                torch_device_name = "cuda"
                if (gpu_index >= 0):
                    torch_device_name = "cuda:" + str(gpu_index)
                print("torch_device_name: " + torch_device_name)
                torch_device = torch.device(torch_device_name)
                model = model.to(torch_device)
                model.eval()
                durations = []
                print(
                    f"Benchmarking Inference {precision} precision type {model_name} "
                )
                for step, img in enumerate(rand_loader):
                    img = getattr(img, precision)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to("cuda"))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= args.WARM_UP:
                        durations.append((end - start) * 1000)
                print(
                    f"{model_name} model average inference time : {sum(durations)/len(durations)}ms"
                )
                del model
                benchmark[model_name] = durations
    return benchmark

f"{platform.uname()}\n{psutil.cpu_freq()}\ncpu_count: {psutil.cpu_count()}\nmemory_available: {psutil.virtual_memory().available}"

if __name__ == "__main__":
    args = parser.parse_args()
    args.BATCH_SIZE *= args.GPU_COUNT

    print("BATCH_SIZE: " + str(args.BATCH_SIZE))
    rand_loader = DataLoader(
        dataset=RandomDataset(args.BATCH_SIZE * (args.WARM_UP + args.NUM_TEST)),
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )
    gpu_count = args.GPU_COUNT
    gpu_index = args.GPU_INDEX
    
    print("gpu_index: " + str(gpu_index))
    print("gpu_count: " + str(gpu_count))

    if (gpu_index >= 0):
        device_name = str(torch.cuda.get_device_name(gpu_index))
    else:
        device_name = str(torch.cuda.get_device_name(0))
    device_name = f"{device_name}"
    if (args.GPU_COUNT > 1):
        device_name = device_name + str(gpu_count) + "X"
    device_name = device_name.replace(" ", "_")
    device_file_name = device_name + "_"
    print("device_name: " + device_name)

    if (gpu_index >= 0):
        folder_name = args.folder + "/" + str(gpu_index) + "/" + device_name
    else:
        folder_name = args.folder + "/" + str(gpu_count) + "X"
    print("folder_name: " + folder_name)

    system_configs = f"{platform.uname()}\n\
                     {psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"
    gpu_configs = [
        gpu_count,
        torch.__version__,
        torch.version.hip,
        torch.version.cuda,
        torch.backends.cudnn.version(),
        device_name,
    ]
    gpu_configs = list(map(str, gpu_configs))
    temp = [
        "GPU_Count: ",
        "Torch_Version : ",
        "ROCM_Version: ",
        "CUDA_Version: ",
        "Cudnn_Version: ",
        "Device_Name: ",
    ]

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    now = datetime.datetime.now()

    start_time = now.strftime("%Y/%m/%d %H:%M:%S")

    print(f"benchmark start : {start_time}")

    for idx, value in enumerate(zip(temp, gpu_configs)):
        gpu_configs[idx] = "".join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name, "system_info.txt"), "w") as f:
        f.writelines(f"benchmark start : {start_time}\n")
        f.writelines("system_configs\n\n")
        f.writelines(system_configs)
        f.writelines("\ngpu_configs\n\n")
        f.writelines(s + "\n" for s in gpu_configs)

    for precision in precisions:
        train_result = train(precision, gpu_index)
        train_result_df = pd.DataFrame(train_result)
        path = f"{folder_name}/{device_file_name}_{precision}_model_train_benchmark.csv"
        train_result_df.to_csv(path, index=False)

        inference_result = inference(precision, gpu_index)
        inference_result_df = pd.DataFrame(inference_result)
        path = f"{folder_name}/{device_file_name}_{precision}_model_inference_benchmark.csv"
        inference_result_df.to_csv(path, index=False)

    now = datetime.datetime.now()

    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"benchmark end : {end_time}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"benchmark end : {end_time}\n")
