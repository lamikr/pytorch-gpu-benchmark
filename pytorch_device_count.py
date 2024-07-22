import os
import torch

device_cnt = torch.cuda.device_count()
#print("device_cnt: " + str(device_cnt))
print(str(device_cnt))
