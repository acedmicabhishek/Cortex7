import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Num GPUs Available:", torch.cuda.device_count())
    print("GPU Devices:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
