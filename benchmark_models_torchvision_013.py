"""Compare speed of different models with batch size 12"""
import torch
import psutil
import datetime
import os
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import json
import torch.multiprocessing
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import platform
import torch.nn as nn
import time

from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from torchvision.models.resnet import (
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
    resnext101_32x8d,
    resnext101_64x4d,
    resnext50_32x4d,
    wide_resnet101_2,
    wide_resnet50_2,
)
from torchvision.models.vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)
from torchvision.models.shufflenetv2 import (
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1

torch.backends.cudnn.benchmark = True
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.


MODEL_LIST = {
    "mnasnet": [mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3],
    "resnet": [
        resnet101,
        resnet152,
        resnet18,
        resnet34,
        resnet50,
        resnext101_32x8d,
        resnext101_64x4d,
        resnext50_32x4d,
        wide_resnet101_2,
        wide_resnet50_2,
    ],
    "densenet": [densenet121, densenet161, densenet169, densenet201],
    "squeezenet": [squeezenet1_0, squeezenet1_1],
    "vgg": [vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn],
    "mobilenetv2": [mobilenet_v2],
    "mobilenetv3": [mobilenet_v3_large, mobilenet_v3_small],
    "shufflenetv2": [
        shufflenet_v2_x0_5,
        shufflenet_v2_x1_0,
        shufflenet_v2_x1_5,
        shufflenet_v2_x2_0,
    ],
}

precisions = ["float", "half", "double"]
# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.
device_name = str(torch.cuda.get_device_name(0))
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
    "--NUM_GPU", "-g", type=int, default=1, required=False, help="Num of gpus"
)
parser.add_argument(
    "--folder",
    "-f",
    type=str,
    default="result",
    required=False,
    help="folder to save results",
)
args = parser.parse_args()
args.BATCH_SIZE *= args.NUM_GPU


class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.randn(3, 224, 224, length)

    def __getitem__(self, index):
        return self.data[:, :, :, index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(
    dataset=RandomDataset(args.BATCH_SIZE * (args.WARM_UP + args.NUM_TEST)),
    batch_size=args.BATCH_SIZE,
    shuffle=False,
    num_workers=8,
)


def train(precision="single"):
    """use fake image for training speed test"""
    target = torch.LongTensor(args.BATCH_SIZE).random_(args.NUM_CLASSES).cuda()
    criterion = nn.CrossEntropyLoss()
    benchmark = {}
    for model_name, models in MODEL_LIST.items():
        for model_funcition in models:
            model = model_funcition(True)
            submodel_name=model_funcition.__str__()
            try:
                submodel_name = submodel_name.replace("<function ", '')
                submodel_name = submodel_name[0 : submodel_name.index(' ')]
            except ValueError:
                pass
            if args.NUM_GPU > 1:
                model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
            model = getattr(model, precision)()
            model = model.to("cuda")
            durations = []
            print(f"Benchmarking Training {precision} precision type {submodel_name} ")
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
            print(f"{submodel_name} model average train time: {sum(durations)/len(durations)} ms")
            del model
            benchmark[submodel_name] = durations
    return benchmark


def inference(precision="float"):
    benchmark = {}
    with torch.no_grad():
        for model_name, models in MODEL_LIST.items():
            for model_funcition in models:
                submodel_name=model_funcition.__str__()
                try:
                    submodel_name = submodel_name.replace("<function ", '')
                    submodel_name = submodel_name[0 : submodel_name.index(' ')]
                except ValueError:
                    pass
                model = model_funcition(True)
                single_module_name = model_funcition()._get_name()
                if args.NUM_GPU > 1:
                    model = nn.DataParallel(model, device_ids=range(args.NUM_GPU))
                model = getattr(model, precision)()
                model = model.to("cuda")
                model.eval()
                durations = []
                print(f"Benchmarking Inference {precision} precision type {submodel_name} ")
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
                    f"{submodel_name} model average inference time: {sum(durations)/len(durations)} ms"
                )
                del model
                benchmark[submodel_name] = durations
    return benchmark


f"{platform.uname()}\n{psutil.cpu_freq()}\ncpu_count: {psutil.cpu_count()}\nmemory_available: {psutil.virtual_memory().available}"


if __name__ == "__main__":
    folder_name = args.folder

    device_name = f"{device_name}_{args.NUM_GPU}_gpus_"
    device_name_und = device_name.replace(" ", "_")
    system_configs = f"{platform.uname()}\n\
                     {psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"
    gpu_configs = [
        torch.cuda.device_count(),
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.cuda.get_device_name(0),
    ]
    gpu_configs = list(map(str, gpu_configs))
    temp = [
        "Number of GPUs on current device : ",
        "CUDA Version : ",
        "Cudnn Version : ",
        "Device Name : ",
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
        train_result = train(precision)
        train_result_df = pd.DataFrame(train_result)
        path = f"{folder_name}/{device_name_und}_{precision}_model_train_benchmark.csv"
        train_result_df.to_csv(path, index=False)

        inference_result = inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path = f"{folder_name}/{device_name_und}_{precision}_model_inference_benchmark.csv"
        inference_result_df.to_csv(path, index=False)

    now = datetime.datetime.now()

    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"benchmark end : {end_time}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"benchmark end : {end_time}\n")
