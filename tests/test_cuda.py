import torch

# Check if CUDA is available and print the device name
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))